import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import torch

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
BOND_TYPES_INV = {v: k for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': 'U',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    AA_NUMBER_NAME = {
        i: k for k, i in AA_NAME_NUMBER.items()
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, device, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        self.residue_id = []  # Store residue ID for each atom
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self.device = device

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            
        # 确保张量都在GPU上
        protein_pos = torch.tensor(np.array(self.pos)).to(self.device)
        dist_matrix = torch.cdist(protein_pos, protein_pos, p=2)
        dist_matrix.fill_diagonal_(float('inf'))
        min_dists, min_indices = torch.min(dist_matrix, dim=1)
        min_dists = min_dists.cpu().numpy()
        min_indices = min_indices.cpu().numpy()
        delete_inds = []
        for i in range(len(min_indices)):
            if self.element[i] != 1: continue
            nearest_atom_indx = min_indices[i]
            if self.element[nearest_atom_indx] == 6 and min_dists[i] < 1.5:
                delete_inds.append(i)
                
         
        # num_atoms = len(self.element)
        # delete_inds = []
        # for i in range(num_atoms):
        #     if self.element[i] == 1:
        #         nearest_atom_indx = i
        #         nearest_atom_dist = 1.5
        #         for j in range(num_atoms):
        #             if self.element[j] != 1 and j!=i:
        #                 dist = np.linalg.norm(self.pos[i] - self.pos[j])
        #                 if dist < nearest_atom_dist:
        #                     nearest_atom_indx = j
        #                     nearest_atom_dist = dist
        #         if nearest_atom_indx == i:
        #             print("did not find an atom join to H!!!")
        #         else:
        #             if self.element[nearest_atom_indx] == 6:
        #                 #delete H join to C
        #                 delete_inds.append(i)
        #             elif self.element[nearest_atom_indx] == 1:
        #                 print("H join to H!!!")
        
        if len(delete_inds) > 0:
            self.atoms = [element for index, element in enumerate(self.atoms) if index not in delete_inds]
            self.element = [element for index, element in enumerate(self.element) if index not in delete_inds]
            self.atomic_weight = [element for index, element in enumerate(self.atomic_weight) if index not in delete_inds]
            self.pos = [element for index, element in enumerate(self.pos) if index not in delete_inds]
            self.atom_name = [element for index, element in enumerate(self.atom_name) if index not in delete_inds]
            self.is_backbone = [element for index, element in enumerate(self.is_backbone) if index not in delete_inds]


        for i, atom in enumerate(self.atoms):
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                atom['res_name'] = 'UNK'
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])
            self.residue_id.append(atom['res_id'])  # Store residue ID for each atom
            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if atom['res_name'] not in self.AA_NAME_NUMBER:
                atom['res_name'] = 'UNK'
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])
            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [i],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(i)
                
        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=int),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=int),
            'residue_id': np.array(self.residue_id, dtype=int)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=int),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id

def remove_carbon_hydrogens(mol):
    """
    此函数会移除所有连接在碳原子上的氢原子，
    同时保留连接在非碳原子（如O、N、S等）上的氢原子。

    参数:
        mol: RDKit Mol 对象（最好已经使用AddHs()添加了显式氢）

    返回:
        一个新的RDKit Mol对象，其中只保留了非碳原子上的氢。
    """
    # 创建一个可编辑的分子对象，以便进行原子操作
    rw_mol = Chem.RWMol(mol)
    # 初始化一个列表，用于记录需要移除的氢原子的索引
    atoms_to_remove = []

    # 遍历分子中的每一个原子
    for atom in rw_mol.GetAtoms():
        # 检查原子是否为氢原子 (原子序数为1)
        if atom.GetAtomicNum() == 1:
            # 获取与当前氢原子相连的原子（通常只有一个）
            neighbors = atom.GetNeighbors()
            if len(neighbors) > 0:
                # 获取连接氢原子的母原子
                parent_atom = neighbors[0]
                # 检查母原子是否是碳原子 (原子序数为6)
                if parent_atom.GetAtomicNum() == 6:
                    # 如果是连接在碳上的氢，则标记为待移除
                    atoms_to_remove.append(atom.GetIdx())
    # 注意：必须从大到小排序索引并反向移除，以防止索引变动导致错误
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx) # 从分子中移除原子

    # 将可编辑分子对象转换回普通的Mol对象并返回
    return rw_mol.GetMol()


def parse_sdf_file(path, add_h=False):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    if isinstance(path, str):
        # read mol
        if path.endswith('.sdf'):
            rdmol = Chem.MolFromMolFile(path, sanitize=False)
        elif path.endswith('.mol2'):
            rdmol = Chem.MolFromMol2File(path, sanitize=False)
        else:
            raise ValueError
    elif isinstance(path, Chem.rdchem.Mol):
        rdmol = path
    else:
        raise ValueError('Unknown type of path: %s' % type(path))
    # Chem.SanitizeMol(rdmol)
    if add_h:
        # Add Hydrogens.
        rdmol = Chem.AddHs(rdmol, addCoords=True)
    rdmol = remove_carbon_hydrogens(rdmol)
    Chem.SanitizeMol(rdmol)

    # rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=int)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=int)
    edge_type = np.array(edge_type, dtype=int)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data
