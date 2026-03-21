import glob
import os,io
import pickle
import random
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import sys
from time import time
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
import traceback
from rdkit import Chem
from torch.utils.data import Subset
import numpy as np
import csv
from typing import Callable, Optional, List, Any




import torch
from torch_geometric.transforms import Compose

from core.datasets.utils import PDBProtein, parse_sdf_file, ATOM_FAMILIES_ID, BOND_TYPES_INV
from core.datasets.pl_data import ProteinLigandData, torchify_dict

import core.utils.transforms as trans
from torch_geometric.data import Batch

class SkipSampleException(Exception):
    pass

class DBReader:
    def __init__(self, path) -> None:
        self.path = path
        self.db = None
        self.keys = None

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.path,
            map_size=30*(1024*1024*1024),   # 100GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __del__(self):
        if self.db is not None:
            self._close_db()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()

        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        # If ligand_filename is not present, construct it from index
        if not hasattr(data, 'ligand_filename') or data.ligand_filename is None:
            data.ligand_filename = f'{data.pdb_id}_{idx}.sdf'
        # assert data.protein_pos.size(0) > 0
        return data

        

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

def write_xyz(filename, coords, elements):
    pass
def process_func(args):
    chunk_id, df_chunk = args
    device_id = chunk_id % 6 + 2 # 假设有6个GPU，循环使用它们,从2号卡开始
    print(f'process_func called: {chunk_id}，{device_id}')
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}') 

    row_list = []
    acc_num = 0
    error_num = 0
    for i, row in df_chunk.iterrows():
        pocket_block = row['cut_protein_pdb']
        ligand_block = row['cut_ligand_sdf']
        #debug NCI info        
        # nci_data = row['nci_info']                                         
        # if isinstance(nci_data, str):
        #     binary_data = bytes.fromhex(nci_data)
        #     nci_data_io = io.BytesIO(binary_data)
        #     pdb_oddt_npy = np.load(nci_data_io, allow_pickle=True)
        # else:
        #     pdb_oddt_npy = pickle.loads(nci_data)

        # pocket_contact_coords = pdb_oddt_npy[0]
        # ligand_contact_coords = pdb_oddt_npy[1]
            

        try:
            pocket_dict = PDBProtein(pocket_block, device=device).to_dict_atom()
            ligand_dict = parse_sdf_file(Chem.MolFromMolBlock(ligand_block, removeHs=False))
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
                device=device
            )
            data.pdb_id = row['pdb_id']
            data.ligand_smiles = row['canonical_smiles']
            data = data.to_dict()  # avoid torch_geometric version issue
            row_list.append(data)
            acc_num += 1
        except Exception as e:
            error_num += 1
            print(f'Skipping ({chunk_id}:{i}) EORROR: {e}')
            print(traceback.format_exc())
            continue
    return chunk_id, row_list, acc_num, error_num

class PocketLigandPairDatasetV2(Dataset):

    def __init__(self, csv_path, transform=None, version='final'):
        super().__init__()
        self.csv_path = csv_path
        self.processed_path = os.path.join(os.path.dirname(self.csv_path),
                                           os.path.basename(self.csv_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def get_files_dir(self):
        """Get the directory where ligand and pocket files are stored."""
        # Get the base name of the CSV file without extension
        base_name = os.path.basename(self.csv_path)
        # Remove .csv extension if present
        if base_name.endswith('.csv'):
            base_name = base_name[:-4]
        # The files directory is in the same directory as the CSV file
        files_dir = os.path.join(os.path.dirname(self.csv_path), f"{base_name}_validation_set")
        return files_dir

    def export_test_files(self, indices=None):
        """
        Export ligand (.sdf) and pocket (.pdb) files for validation/test sets.
        Files are stored in a directory named after the dataset.

        Args:
            indices: List of indices to export. If None, export all samples.
        """
        files_dir = self.get_files_dir()
        os.makedirs(files_dir, exist_ok=True)

        if indices is None:
            indices = range(len(self))

        print(f"Exporting ligand and pocket files to {files_dir}...")

        for idx in tqdm(indices, desc="Exporting files"):
            data = self[idx]
            ligand_filename = getattr(data, 'ligand_filename', f'{data.pdb_id}_{idx}.sdf')
            # Export ligand SDF file
            ligand_path = os.path.join(files_dir, ligand_filename)
            if not os.path.exists(ligand_path):
                # Check if ligand_sdf_block exists in data
                if hasattr(data, 'ligand_sdf_block'):
                    sdf_block = data.ligand_sdf_block
                else:
                    # Generate SDF from ligand_pos and ligand_element
                    from rdkit import Chem
                    from rdkit.Chem import AllChem

                    # Create RDKit mol from positions and elements
                    mol = Chem.RWMol()
                    ligand_pos = data.ligand_pos.cpu().numpy()
                    ligand_element = data.ligand_element.cpu().numpy()

                    # Add atoms
                    for i, (pos, elem_num) in enumerate(zip(ligand_pos, ligand_element)):
                        atom = Chem.Atom(int(elem_num))
                        mol.AddAtom(atom)

                    # Add conformer
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for i, pos in enumerate(ligand_pos):
                        # Ensure pos is a 3D tuple and doesn't contain NaN/Inf
                        pos_tuple = tuple(float(x) for x in pos)
                        if len(pos_tuple) != 3:
                            raise ValueError(f"Invalid position shape {pos.shape}, expected 3D coordinates")
                        if any(np.isnan(x) or np.isinf(x) for x in pos_tuple):
                            raise ValueError(f"Invalid coordinates (NaN/Inf): {pos_tuple}")
                        conf.SetAtomPosition(i, pos_tuple)
                    # print(f"ligand_pos shape: {ligand_pos.shape}")  # 应该是 (N, 3)
                    # print(f"ligand_pos sample: {ligand_pos[0]}")    # 检查第一个坐标
                    # print(f"Contains NaN: {np.isnan(ligand_pos).any()}")
                    # print(f"Contains Inf: {np.isinf(ligand_pos).any()}")
                    mol.AddConformer(conf)

                    # Add bonds if available
                    if hasattr(data, 'ligand_bond_index') and hasattr(data, 'ligand_bond_type'):
                        bond_index = data.ligand_bond_index.cpu().numpy()
                        bond_type = data.ligand_bond_type.cpu().numpy()
                        for i in range(bond_index.shape[1]):
                            start = int(bond_index[0, i])
                            end = int(bond_index[1, i])
                            # Skip duplicate edges (bond_index stores each edge twice)
                            if start >= end:
                                continue
                            btype = int(bond_type[i])
                            if btype > 0 and btype <= 4:
                                mol.AddBond(start, end, BOND_TYPES_INV[btype])

                    mol = mol.GetMol()
                    sdf_block = Chem.MolToMolBlock(mol,  kekulize=False)

                with open(ligand_path, 'w') as f:
                    f.write(sdf_block)

            # Export pocket PDB file
            pocket_filename = ligand_filename.replace('.sdf', '.pdb')
            pocket_path = os.path.join(files_dir, pocket_filename)

            if not os.path.exists(pocket_path):
                # Check if cut_protein_pdb block exists
                if hasattr(data, 'cut_protein_pdb'):
                    pdb_block = data.cut_protein_pdb
                elif hasattr(data, 'pocket_pdb_block'):
                    pdb_block = data.pocket_pdb_block
                else:
                    # Generate PDB from protein_pos and protein_element
                    protein_pos = data.protein_pos.cpu().numpy()
                    protein_element = data.protein_element.cpu().numpy()
                    protein_atom_name = getattr(data, 'protein_atom_name', None)
                    protein_is_backbone = getattr(data, 'protein_is_backbone', None)
                    protein_atom_to_aa_type = getattr(data, 'protein_atom_to_aa_type', None)
                    protein_residue_id = getattr(data, 'protein_residue_id', None)

                    # Import AA_NUMBER_NAME from utils
                    from core.datasets.utils import PDBProtein
                    AA_NUMBER_NAME = PDBProtein.AA_NUMBER_NAME

                    pdb_lines = ["HEADER    POCKET"]

                    # Track residue ID assignment
                    current_res_id = 1
                    prev_aa_type = None

                    for i, (pos, elem_num) in enumerate(zip(protein_pos, protein_element)):
                        elem_sym = Chem.GetPeriodicTable().GetElementSymbol(int(elem_num))
                        atom_name = protein_atom_name[i] if protein_atom_name is not None else f"{'C' if elem_sym == 'C' else 'X'}  "
                        is_backbone = protein_is_backbone[i] if protein_is_backbone is not None else False

                        # Get residue name from atom_to_aa_type if available
                        if protein_atom_to_aa_type is not None:
                            aa_type_idx = int(protein_atom_to_aa_type[i])
                            res_name = AA_NUMBER_NAME.get(aa_type_idx, 'UNK')

                            # Auto-increment residue ID when amino acid type changes
                            if protein_residue_id is None:
                                if prev_aa_type is not None and aa_type_idx != prev_aa_type:
                                    current_res_id += 1
                                prev_aa_type = aa_type_idx
                                res_id = current_res_id
                            else:
                                res_id = protein_residue_id[i]
                        else:
                            res_name = "UNK"
                            res_id = 1

                        # Format PDB line
                        record_type = "ATOM"
                        atom_id = i + 1
                        atom_name_formatted = f"{atom_name:>4}"
                        chain = "A"
                        x, y, z = pos
                        occupancy = 1.0
                        temp_factor = 0.0
                        element = f"{elem_sym:>2}"
                        charge = ""

                        line = f"{record_type:<6}{atom_id:>5} {atom_name_formatted} {res_name:>3} {chain:1}{res_id:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{temp_factor:>6.2f}          {element:<2}{charge}"
                        pdb_lines.append(line)
                    pdb_lines.append("END")
                    pdb_block = "\n".join(pdb_lines)

                with open(pocket_path, 'w') as f:
                    f.write(pdb_block)

        print(f"Exported {len(indices)} ligand and pocket files to {files_dir}")
        return files_dir
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=100*(1024*1024*1024),   # 100GB
            create=True,
            subdir=False,
           
            readonly=False,  # Writable
        )
        csv_chunk_reader = pd.read_csv(self.csv_path,chunksize=1000)
        
        # # For debug:
        # for args in enumerate(csv_chunk_reader):
        #     result = process_func(args)  # warm u
        #     chunk_id, row_list, acc_num, error_num = result
        #     with db.begin(write=True, buffers=True) as txn:
        #         for i, data in enumerate(row_list):
        #             idx = chunk_id * 2000 + i
        #             txn.put(
        #                 key=str(idx).encode(),
        #                 value=pickle.dumps(data)
        #                 )
                        
        #         print(f"Chunk {chunk_id} Done! row num:{acc_num}, error num: {error_num}")
    
        with Pool(processes=6) as pool:
            results = pool.imap(process_func, enumerate(csv_chunk_reader))
            for result in results:
                chunk_id, row_list, acc_num, error_num = result
                print(f'Got result from process_func: {chunk_id}, {acc_num}, {error_num}')

                with db.begin(write=True) as txn:
                    for i, data in enumerate(row_list):
                        idx = chunk_id * 1000 + i
                        txn.put(
                            key=str(idx).encode(),
                            value=pickle.dumps(data)
                            )     
                print(f"Chunk {chunk_id} Done! row num:{acc_num}, error num: {error_num}")   
            
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def make_test_split(self, split_ratio=0.9, export_files=True):
        """
        Create a train/test split of the dataset.
        when split_ratio is greater than 1, it is treated as the absolute number of testing samples.

        Args:
            split_ratio: Ratio for train/test split (0.9 means 90% train, 10% test)
            export_files: If True, export ligand and pocket files for the test set

        Returns:
            dict with 'train' and 'test' Subsets
        """
        total_size = len(self)
        indices = list(range(total_size))
        #log the time cost:
        print("make_test_split init random indices...")
        indices = random.sample(indices, total_size)  # shuffle indices
        assert split_ratio >= 0.0, "split_ratio must >= 0"
        if split_ratio <= 1.0:
            split = int(total_size * split_ratio)
        else:
            split = total_size - int(split_ratio)
        print("make_test_split init random indices done!")
        train_indices = indices[:split]
        test_indices = indices[split:]

        result = {
            'train': Subset(self, train_indices),
            'test': Subset(self, test_indices)
        }

        # Export files for test set if requested
        if export_files:
            print("Exporting ligand and pocket files for test set...")
            self.export_test_files(test_indices)

        return result

class PocketLigandGeneratedPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='4-decompdiff'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.generated_path = os.path.join('./data/all_results', f'{version}_docked_pose_checked.pt')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.generated_path, 'rb') as f:
            results = torch.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            idx = -1
            for i, res in tqdm(enumerate(results), total=len(results)):
                if isinstance(res, dict):
                    res = [res]
                for r in res:
                    idx += 1
                    mol = r["mol"]
                    ligand_fn = r["ligand_filename"]
                    pocket_fn = os.path.join(
                        os.path.dirname(ligand_fn),
                        os.path.basename(ligand_fn)[:-4] + '_pocket10.pdb'
                    )

                    if pocket_fn is None: continue
                    try:
                        data_prefix = self.raw_path
                        pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                        ligand_dict = parse_sdf_file(mol)
                        # ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                        data = ProteinLigandData.from_protein_ligand_dicts(
                            protein_dict=torchify_dict(pocket_dict),
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        data.protein_filename = pocket_fn
                        data.ligand_filename = ligand_fn
                        data = data.to_dict()  # avoid torch_geometric version issue
                        txn.put(
                            key=str(idx).encode(),
                            value=pickle.dumps(data)
                        )
                    except Exception as e:
                        num_skipped += 1
                        print('Skipping (%d) %s' % (num_skipped, ligand_fn, ), e)
                        continue
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class PocketLigandPairDatasetFromComplex(Dataset):
    def __init__(self, raw_path, transform=None, version='final', radius=10.0):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        base_name = os.path.basename(self.raw_path)
        if 'pocket' in base_name:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        else:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                            os.path.basename(self.raw_path) + f'_pocket{radius}_processed_{version}.lmdb')
        self.transform = transform
        self.reader = DBReader(self.processed_path)

        self.radius = radius

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 50GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
            max_readers=256,
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        print('Processing data...', 'index', self.index_path, index[0])

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    # clip pocket
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    protein = PDBProtein(os.path.join(data_prefix, pocket_fn))
                    selected = protein.query_residues_ligand(ligand_dict, self.radius)
                    pdb_block_pocket = protein.residues_to_pdb_block(selected)
                    pocket_dict = PDBProtein(pdb_block_pocket).to_dict_atom()

                    # pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    # ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ), e)
                    with open('skipped.txt', 'a') as f:
                        f.write('Skip %s due to %s\n' % (ligand_fn, e))
                    continue
        db.close()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    

class PocketLigandPairDatasetFeaturized(Dataset):
    def __init__(self, raw_path, ligand_atom_mode, version='simple'):
        """
        in simple version, only these features are saved for better IO:
            protein_pos, protein_atom_feature, protein_element, 
            ligand_pos, ligand_atom_feature_full, ligand_element
        """
        self.raw_path = raw_path
        self.ligand_atom_mode = ligand_atom_mode
        self.version = version

        if version == 'simple':
            self.features_to_save = [
                'protein_pos', 'protein_atom_feature', 'protein_element', 
                'ligand_pos', 'ligand_atom_feature_full', 'ligand_element',
                'protein_filename', 'ligand_filename',
            ]
        else:
            raise NotImplementedError

        self.transformed_path = os.path.join(
            os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + 
            f'_{ligand_atom_mode}_transformed_{version}.pt'
        )
        if not os.path.exists(self.transformed_path):
            print(f'{self.transformed_path} does not exist, begin transforming data')
            self._transform()
        else:
            print(f'reading data from {self.transformed_path}...')
            tic = time()
            tr_data = torch.load(self.transformed_path)
            toc = time()
            print(f'{toc - tic} elapsed')
            self.train_data, self.test_data = tr_data['train'], tr_data['test']
            self.protein_atom_feature_dim = tr_data['protein_atom_feature_dim']
            self.ligand_atom_feature_dim = tr_data['ligand_atom_feature_dim']
        
    def _transform(self):
        raw_dataset = PocketLigandPairDataset(self.raw_path, None, 'final')

        split_path = os.path.join(
            os.path.dirname(self.raw_path), 'crossdocked_pocket10_pose_split.pt',
        )
        split = torch.load(split_path)
        train_ids, test_ids = split['train'], split['test']
        print(f'train_size: {len(train_ids)}, test_size: {len(test_ids)}')

        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(self.ligand_atom_mode)
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            # trans.FeaturizeLigandBond(),
        ]
        transform = Compose(transform_list)
        self.protein_atom_feature_dim = protein_featurizer.feature_dim
        self.ligand_atom_feature_dim = ligand_featurizer.feature_dim

        def _transform_subset(ids):
            data_list = []

            for idx in tqdm(ids):
                data = raw_dataset[idx]
                data = transform(data)
                tr_data = {}
                for k in self.features_to_save:
                    tr_data[k] = getattr(data, k)
                tr_data['id'] = idx
                tr_data = ProteinLigandData(**tr_data)
                data_list.append(tr_data)
            return data_list

        self.train_data = _transform_subset(train_ids)
        print(f'train_size: {len(self.train_data)}, {sys.getsizeof(self.train_data)}')
        self.test_data = _transform_subset(test_ids)
        print(f'test_size: {len(self.test_data)}, {sys.getsizeof(self.test_data)}')
        torch.save({
            'train': self.train_data, 'test': self.test_data,
            'protein_atom_feature_dim': self.protein_atom_feature_dim,
            'ligand_atom_feature_dim': self.ligand_atom_feature_dim,
        }, self.transformed_path)

class ChunkedCSVDataset(Dataset):
    def __init__(self, 
                 file_path: str, 
                 chunk_size: int = 10000,
                 max_lines: Optional[int] = None,   
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 usecols: Optional[List] = None,
                 dtype: Optional[dict] = None):
        """
        基于Pandas分块读取的大型CSV数据集
        
        参数:
            file_path: CSV文件路径
            chunk_size: 每个分块的行数
            transform: 特征转换函数
            target_transform: 标签转换函数
            usecols: 指定要读取的列[3](@ref)
            dtype: 列数据类型优化[3](@ref)
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.transform = transform
        self.target_transform = target_transform
        self.usecols = usecols
        self.dtype = dtype
        self.max_lines = max_lines
        
        # 获取总行数(排除标题行)
        if self.max_lines is not None and self.max_lines > 0:
            self.total_rows = self.max_lines
        else:
            self.total_rows = self._count_lines()
        
        # 预计算每个分块的起始索引
        self.chunk_ranges = []
        for i in range(0, self.total_rows, chunk_size):
            end = min(i + chunk_size, self.total_rows)
            self.chunk_ranges.append((i, end))
        
        # 当前加载的分块
        self.current_chunk = None
        self.current_chunk_idx = -1
        self.current_chunk_start_idx = 0
        
        # 获取列名
        self.column_names = self._get_column_names()
        
    def _count_lines(self) -> int:
        """计算CSV文件行数(排除标题行)"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            f.readline()
            return sum(1 for _ in f)
    
    def _get_column_names(self) -> List[str]:
        """获取CSV文件的列名"""
        # 读取第一行获取列名
        sample_df = pd.read_csv(self.file_path, nrows=1)
        return list(sample_df.columns)
    
    def _load_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """加载指定分块的数据"""
        if chunk_idx < 0 or chunk_idx >= len(self.chunk_ranges):
            raise IndexError(f"Chunk index {chunk_idx} out of range")
        
        start_idx, end_idx = self.chunk_ranges[chunk_idx]
        
        # 计算需要跳过的行数(包括标题行)
        skiprows = start_idx + 1
        
        # 计算需要读取的行数
        nrows = end_idx - start_idx
        
        # 使用pandas分块读取特定范围的数据[1,7](@ref)
        chunk = pd.read_csv(
            self.file_path,
            skiprows=skiprows,
            nrows=nrows,
            header=None,  # 因为我们已经跳过了标题行
            names=self.column_names,
            usecols=self.usecols,
            dtype=self.dtype
        )
        
        return chunk
    
    def __len__(self) -> int:
        return self.total_rows
    
    def __getitem__(self, idx: int) -> Any:
        """获取单个样本"""
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range [0, {self.total_rows-1}]")
        
        # 计算当前样本所在的分块
        chunk_idx = idx // self.chunk_size
        
        # 如果分块未加载，加载对应分块
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk = self._load_chunk(chunk_idx)
            self.current_chunk_idx = chunk_idx
            self.current_chunk_start_idx = chunk_idx * self.chunk_size
        
        # 获取分块内的相对索引
        local_idx = idx - self.current_chunk_start_idx
        
        # 获取行数据
        row = self.current_chunk.iloc[local_idx]
        
        # 假设最后一列是标签，其余是特征
        sdf_str = row['conf']
        try:
            mol = Chem.MolFromMolBlock(sdf_str, removeHs=False)
            ligand_dict = parse_sdf_file(mol)
            data = ProteinLigandData.from_protein_ligand_dicts(
                        ligand_dict=torchify_dict(ligand_dict)
                    )
            data.ligand_filename = row['uni_id']

            if self.transform:
                data = self.transform(data)
            return data
        except Exception as e:
            print(f"Error processing SDF for index {idx}: {e}\n{row}\n")
            return self.__getitem__(random.randint(0, idx-1))
            # raise SkipSampleException(f"Error loading data at index {idx}")
 
 
def ligand_process_func(args):
    chunk_id, df_chunk = args
    print(f'process_func called: {chunk_id}')

    row_list = []
    acc_num = 0
    error_num = 0
    for i, row in df_chunk.iterrows():
        ligand_block = row['conf']
        #debug NCI info        
        # nci_data = row['nci_info']                                         
        # if isinstance(nci_data, str):
        #     binary_data = bytes.fromhex(nci_data)
        #     nci_data_io = io.BytesIO(binary_data)
        #     pdb_oddt_npy = np.load(nci_data_io, allow_pickle=True)
        # else:
        #     pdb_oddt_npy = pickle.loads(nci_data)

        # pocket_contact_coords = pdb_oddt_npy[0]
        # ligand_contact_coords = pdb_oddt_npy[1]
            

        try:
            ligand_dict = parse_sdf_file(Chem.MolFromMolBlock(ligand_block, removeHs=False),add_h=False)
            data = ProteinLigandData.from_protein_ligand_dicts(
                ligand_dict=torchify_dict(ligand_dict)
            )
            data.ligand_filename = row['uni_id']
            data.ligand_smiles = row['smi']
            data.ligand_sdf_block = ligand_block
            data = data.to_dict()  # avoid torch_geometric version issue
            row_list.append(data)
            acc_num += 1
        except Exception as e:
            error_num += 1
            print(f'Skipping ({chunk_id}:{i}) EORROR: {e}')
            print(traceback.format_exc())
            continue
    return chunk_id, row_list, acc_num, error_num

class LigandDataset(Dataset):

    def __init__(self, csv_path, transform=None, version='final'):
        super().__init__()
        self.csv_path = csv_path
        self.transform = transform
        self.processed_path = os.path.join(os.path.dirname(self.csv_path),
                                           os.path.basename(self.csv_path) + f'_processed_{version}.lmdb')
        self.reader = DBReader(self.processed_path)

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=100*(1024*1024*1024),   # 100GB
            create=True,
            subdir=False,
           
            readonly=False,  # Writable
        )
        csv_chunk_reader = pd.read_csv(self.csv_path,chunksize=1000)
        
        # # For debug:
        # for args in enumerate(csv_chunk_reader):
        #     result = process_func(args)  # warm u
        #     chunk_id, row_list, acc_num, error_num = result
        #     with db.begin(write=True, buffers=True) as txn:
        #         for i, data in enumerate(row_list):
        #             idx = chunk_id * 2000 + i
        #             txn.put(
        #                 key=str(idx).encode(),
        #                 value=pickle.dumps(data)
        #                 )
                        
        #         print(f"Chunk {chunk_id} Done! row num:{acc_num}, error num: {error_num}")
    
        with Pool(processes=10) as pool:
            results = pool.imap(ligand_process_func, enumerate(csv_chunk_reader))
            for result in results:
                chunk_id, row_list, acc_num, error_num = result

                with db.begin(write=True) as txn:
                    for i, data in enumerate(row_list):
                        idx = chunk_id * 1000 + i
                        txn.put(
                            key=str(idx).encode(),
                            value=pickle.dumps(data)
                            )     
                print(f"Chunk {chunk_id} Done! row num:{acc_num}, error num: {error_num}")   
            
        db.close()
    
    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        data = self.reader[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def get_range(self, start_idx=0, size=None):
        total_size = len(self)
        assert start_idx < total_size, "start index can not greater than total size!!"
        if size is None or size < 1:
            size = total_size
        end_idx = start_idx + size - 1
        if end_idx >= total_size:
            end_idx = total_size - 1
        indices = list(range(start_idx, end_idx+1))
        return Subset(self, indices)

def load_pocket_size_dict(dirpath, pattern=""):
    pocket_size_dict = {}
    pocket_size_files = glob.glob(f"{dirpath}/**/{pattern}", recursive=True)

    # for path in os.listdir(dirpath):
    for path in pocket_size_files:
        # path = os.path.join(dirpath, path)
        print(f'Loading pocket size from {path}...')
        with open(path, 'r') as f:
            total_lines = sum(1 for line in f) # 或者使用 len(f.readlines())，但会多读一次文件
            f.seek(0)  # 将文件指针重置回开头
            # 使用 tqdm 包装文件读取过程
            for row in tqdm(f.readlines(), total=total_lines, desc="Load pocket size dict", unit="line"):
                id, smi, size = row.strip().split('\t')
                pocket_size_dict[id] = int(float(size))
    return pocket_size_dict
    
def sample_pocket_nums(pocket_num, sample_ratios=[-1,0,1,2]):
    pocket_num_min = 200
    pocket_num_max = 800
    pocket_nums = []  # Sample several numbers around pocket_num, interval is 30
    for i in sample_ratios:
        v = pocket_num + i * 30
        if v < pocket_num_min:
            v = 200
        if v > pocket_num_max:
            v = 800
        pocket_nums.append(v)
    return pocket_nums
        
if __name__ == '__main__':
    # original dataset
    # dataset = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')
    # print(len(dataset), dataset[0])

    # dataset = PocketLigandPairDatasetV2('/home/jovyan/data/EDMol/complex_ext_v2.csv')
    # mp.set_start_method('spawn')
    # dataset = PocketLigandPairDatasetV2('/home/jovyan/data/EDMol/complex_rcomplexdb_273w/complex_ext_v2.csv')
    # dataset = LigandDataset('/home/jovyan/data/EDMol/pretrain_592.csv')
    dataset = LigandDataset('/home/jovyan/data/EDMol/pretrain_108w.csv')
    print(len(dataset), dataset[0])
    print(dataset[0].ligand_element)
    print(dataset[0].ligand_pos)

    ############################################################

    # test DecompDiffDataset
    # dataset = PocketLigandGeneratedPairDataset('./data/crossdocked_pocket10')
    # print(len(dataset), dataset[0])

    ############################################################

    # test featurized dataset (GPU accelerated)
    # path = './data/crossdocked_v1.1_rmsd1.0_pocket10'
    # ligand_atom_mode = 'add_aromatic'

    # dataset = PocketLigandPairDatasetFeaturized(path, ligand_atom_mode)
    # train_data, test_data = dataset.train_data, dataset.test_data
    # print(f'train_size: {len(train_data)}, {sys.getsizeof(train_data)}')
    # print(f'test_size: {len(test_data)}, {sys.getsizeof(test_data)}')
    # print(test_data[0], sys.getsizeof(test_data[0]))

    ############################################################

    # test featurization
    # find all atom types
    # atom_types = {(1, False): 0}

    # dataset = PocketLigandPairDataset(path, transform)
    # for i in tqdm(range(len(dataset))):
    #     data = dataset[i]
    #     element_list = data.ligand_element
    #     hybridization_list = data.ligand_hybridization
    #     aromatic_list = [v[trans.AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]

    #     types = [(e, a) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
    #     for t in types:
    #         t = (t[0].item(), bool(t[1].item()))
    #         if t not in atom_types:
    #             atom_types[t] = 0
    #         atom_types[t] += 1

    # idx = 0
    # for k in sorted(atom_types.keys()):
    #     print(f'{k}: {idx}, # {atom_types[k]}')
    #     idx += 1

    ############################################################
    
    # count atom types
    # type_counter, aromatic_counter, full_counter = {}, {}, {}
    # for i, data in enumerate(tqdm(dataset)):
    #     element_list = data.ligand_element
    #     hybridization_list = data.ligand_hybridization
    #     aromatic_list = [v[trans.AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
    #     flag = False

    #     for atom_type in element_list:
    #         atom_type = int(atom_type.item())
    #         if atom_type not in type_counter:
    #             type_counter[atom_type] = 0
    #         type_counter[atom_type] += 1

    #     for e, a in zip(element_list, aromatic_list):
    #         e = int(e.item())
    #         a = bool(a.item())
    #         key = (e, a)
    #         if key not in aromatic_counter:
    #             aromatic_counter[key] = 0
    #         aromatic_counter[key] += 1

    #         if key not in trans.MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
    #             flag = True

    #     for e, h, a in zip(element_list, hybridization_list, aromatic_list):
    #         e = int(e.item())
    #         a = bool(a.item())
    #         key = (e, h, a)
    #         if key not in full_counter:
    #             full_counter[key] = 0
    #         full_counter[key] += 1
        
    # print('type_counter', type_counter)
    # print('aromatic_counter', aromatic_counter)
    # print('full_counter', full_counter)



