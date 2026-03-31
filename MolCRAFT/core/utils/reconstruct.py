"""
https://github.com/mattragoza/liGAN/blob/master/fitting.py

License: GNU General Public License v2.0
https://github.com/mattragoza/liGAN/blob/master/LICENSE
"""
import itertools

import numpy as np
from rdkit.Chem import AllChem
from rdkit import Geometry
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class MolReconError(Exception):
    pass


def reachable_r(a, b, seenbonds):
    '''Recursive helper.'''

    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr, b, seenbonds):
                return True
    return False


def reachable(a, b):
    '''Return true if atom b is reachable from a without using the bond between them.'''
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False  # this is the _only_ bond for one atom
    # otherwise do recursive traversal
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a, b, seenbonds)


def forms_small_angle(a, b, cutoff=60):
    '''Return true if bond between a and b is part of a small angle
    with a neighbor of a only.'''

    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            degrees = b.GetAngle(a, nbr)
            if degrees < cutoff:
                return True
    return False


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for xyz, t in zip(xyz, atomic_numbers):
        x, y, z = xyz
        # ch = struct.channels[t]
        atom = mol.NewAtom()
        atom.SetAtomicNum(t)
        atom.SetVector(x, y, z)
        atoms.append(atom)
    return mol, atoms


def connect_the_dots(mol, atoms, indicators, covalent_factor=1.3):
    '''Custom implementation of ConnectTheDots.  This is similar to
    OpenBabel's version, but is more willing to make long bonds 
    (up to maxbond long) to keep the molecule connected.  It also 
    attempts to respect atom type information from struct.
    atoms and struct need to correspond in their order
    Assumes no hydrogens or existing bonds.
    '''

    """
    for now, indicators only include 'is_aromatic'
    """
    pt = AllChem.GetPeriodicTable()

    if len(atoms) == 0:
        return

    mol.BeginModify()

    # just going to to do n^2 comparisons, can worry about efficiency later
    coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))
    # types = [struct.channels[t].name for t in struct.c]

    for i, j in itertools.combinations(range(len(atoms)), 2):
        a = atoms[i]
        b = atoms[j]
        a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
        b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
        if dists[i, j] < a_r + b_r:
            flag = 0
            if indicators and indicators[i] and indicators[j]:
                flag = ob.OB_AROMATIC_BOND
            mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)

    atom_maxb = {}
    for (i, a) in enumerate(atoms):
        # set max valance to the smallest max allowed by openbabel or rdkit
        # since we want the molecule to be valid for both (rdkit is usually lower)
        maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))

        if a.GetAtomicNum() == 16:  # sulfone check
            if count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6

        # if indicators[i][ATOM_FAMILIES_ID['Donor']]:
        #     maxb -= 1 #leave room for hydrogen
        # if 'Donor' in types[i]:
        #     maxb -= 1 #leave room for hydrogen
        atom_maxb[a.GetIdx()] = maxb

    # remove any impossible bonds between halogens
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        '''Return bonds sorted by their distortion'''
        bonds = [b for b in biter]
        binfo = []
        for bond in bonds:
            bdist = bond.GetLength()
            # compute how far away from optimal we are
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
            stretch = bdist / ideal
            binfo.append((stretch, bond))
        binfo.sort(reverse=True, key=lambda t: t[0])  # most stretched bonds first
        return binfo

    binfo = get_bond_info(ob.OBMolBondIter(mol))
    # now eliminate geometrically poor bonds
    for stretch, bond in binfo:

        # can we remove this bond without disconnecting the molecule?
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        # as long as we aren't disconnecting, let's remove things
        # that are excessively far away (0.45 from ConnectTheDots)
        # get bonds to be less than max allowed
        # also remove tight angles, because that is what ConnectTheDots does
        if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):
            # don't fragment the molecule
            if not reachable(a1, a2):
                continue
            mol.DeleteBond(bond)

    # prioritize removing hypervalency causing bonds, do more valent
    # constrained atoms first since their bonds introduce the most problems
    # with reachability (e.g. oxygen)
    hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
    hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
    for mb, diff, a in hypers:
        if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
            continue
        binfo = get_bond_info(ob.OBAtomBondIter(a))
        for stretch, bond in binfo:

            if stretch < 0.9:  # the two atoms are too closed to remove the bond
                continue
            # can we remove this bond without disconnecting the molecule?
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # get right valence
            if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                # don't fragment the molecule
                if not reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)
                if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                    break  # let nbr atoms choose what bonds to throw out

    mol.EndModify()


def convert_ob_mol_to_rd_mol(ob_mol, struct=None):
    '''Convert OBMol to RDKit mol, fixing up issues'''
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = AllChem.RWMol()
    rd_conf = AllChem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = AllChem.Atom(ob_atom.GetAtomicNum())
        # TODO copy format charge
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            # don't commit to being aromatic unless rdkit will be okay with the ring status
            # (this can happen if the atoms aren't fit well enough)
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        ob_coords = ob_atom.GetVector()
        x = ob_coords.GetX()
        y = ob_coords.GetY()
        z = ob_coords.GetZ()
        rd_coords = Geometry.Point3D(x, y, z)
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx() - 1
        j = ob_bond.GetEndAtomIdx() - 1
        bond_order = ob_bond.GetBondOrder()
        if bond_order == 1:
            rd_mol.AddBond(i, j, AllChem.BondType.SINGLE)
        elif bond_order == 2:
            rd_mol.AddBond(i, j, AllChem.BondType.DOUBLE)
        elif bond_order == 3:
            rd_mol.AddBond(i, j, AllChem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(bond_order))

        if ob_bond.IsAromatic():
            bond = rd_mol.GetBondBetweenAtoms(i, j)
            bond.SetIsAromatic(True)

    rd_mol = AllChem.RemoveHs(rd_mol, sanitize=False)

    pt = AllChem.GetPeriodicTable()
    # if double/triple bonds are connected to hypervalent atoms, decrement the order

    # TODO: fix seg fault
    # if struct is not None:
    #     positions = struct
    positions = rd_mol.GetConformer().GetPositions()
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == AllChem.BondType.DOUBLE or bond.GetBondType() == AllChem.BondType.TRIPLE:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # TODO: ugly fix
            dist = np.linalg.norm(positions[i] - positions[j])
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d, bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = AllChem.BondType.SINGLE
            if bond.GetBondType() == AllChem.BondType.TRIPLE:
                btype = AllChem.BondType.DOUBLE
            bond.SetBondType(btype)

    # fix up special cases
    for atom in rd_mol.GetAtoms():
        # set nitrogens with 4 neighbors to have a charge
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

        # check if there are any carbon atoms with 2 double C-C bonds
        # if so, convert one to a single bond
        if atom.GetAtomicNum() == 6 and atom.GetDegree() == 4:
            cnt = 0
            i = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6:
                    j = nbr.GetIdx()
                    bond = rd_mol.GetBondBetweenAtoms(i, j)
                    if bond.GetBondType() == AllChem.BondType.DOUBLE:
                        cnt += 1
            if cnt == 2:
                for nbr in atom.GetNeighbors():
                    if nbr.GetAtomicNum() == 6:
                        j = nbr.GetIdx()
                        bond = rd_mol.GetBondBetweenAtoms(i, j)
                        if bond.GetBondType() == AllChem.BondType.DOUBLE:
                            bond.SetBondType(AllChem.BondType.SINGLE)
                            break

    rd_mol = AllChem.AddHs(rd_mol, addCoords=True)
    # TODO: fix seg fault
    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        pos = positions[i]
        if not np.all(np.isfinite(pos)):
            # hydrogens on C fragment get set to nan (shouldn't, but they do)
            rd_mol.GetConformer().SetAtomPosition(i, center)

    try:
        AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
    except:
        raise MolReconError()
    # try:
    #     AllChem.SanitizeMol(rd_mol,AllChem.SANITIZE_ALL^AllChem.SANITIZE_KEKULIZE)
    # except: # mtr22 - don't assume mols will pass this
    #     pass
    #     # dkoes - but we want to make failures as rare as possible and should debug them
    #     m = pybel.Molecule(ob_mol)
    #     i = np.random.randint(1000000)
    #     outname = 'bad%d.sdf'%i
    #     print("WRITING",outname)
    #     m.write('sdf',outname,overwrite=True)
    #     pickle.dump(struct,open('bad%d.pkl'%i,'wb'))

    # but at some point stop trying to enforce our aromaticity -
    # openbabel and rdkit have different aromaticity models so they
    # won't always agree.  Remove any aromatic bonds to non-aromatic atoms
    for bond in rd_mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol


def calc_valence(rdatom):
    '''Can call GetExplicitValence before sanitize, but need to
    know this to fix up the molecule to prevent sanitization failures'''
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def count_nbrs_of_elem(atom, atomic_num):
    '''
    Count the number of neighbors atoms
    of atom with the given atomic_num.
    '''
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def fixup(atoms, mol, indicators):
    '''Set atom properties to match channel.  Keep doing this
    to beat openbabel over the head with what we want to happen.'''

    """
    for now, indicators only include 'is_aromatic'
    """
    mol.SetAromaticPerceived(True)  # avoid perception
    for i, atom in enumerate(atoms):
        # ch = struct.channels[t]
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)

        # if ind[ATOM_FAMILIES_ID['Donor']]:
        #     if atom.GetExplicitDegree() == atom.GetHvyDegree():
        #         if atom.GetHvyDegree() == 1 and atom.GetAtomicNum() == 7:
        #             atom.SetImplicitHCount(2)
        #         else:
        #             atom.SetImplicitHCount(1) 

        # elif ind[ATOM_FAMILIES_ID['Acceptor']]: # NOT AcceptorDonor because of else
        #     atom.SetImplicitHCount(0)   

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():  # Nitrogen, Oxygen
            # this is a little iffy, ommitting until there is more evidence it is a net positive
            # we don't have aromatic types for nitrogen, but if it
            # is in a ring with aromatic carbon mark it aromatic as well
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def raw_obmol_from_generated(data):
    xyz = data.ligand_context_pos.clone().cpu().tolist()
    atomic_nums = data.ligand_context_element.clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    return mol, atoms


UPGRADE_BOND_ORDER = {AllChem.BondType.SINGLE: AllChem.BondType.DOUBLE, AllChem.BondType.DOUBLE: AllChem.BondType.TRIPLE}


def postprocess_rd_mol_1(rdmol):
    rdmol = AllChem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in nbh_list:
            nbh_list[begin] = [end]
        else:
            nbh_list[begin].append(end)

        if end not in nbh_list:
            nbh_list[end] = [begin]
        else:
            nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list[idx]:
                if j <= idx: continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = AllChem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol


def reconstruct_from_generated_backup(xyz, atomic_nums, aromatic=None, basic_mode=True):
    """
    will utilize data.ligand_pos, data.ligand_element, data.ligand_atom_feature_full to reconstruct mol
    """
    # xyz = data.ligand_pos.clone().cpu().tolist()
    # atomic_nums = data.ligand_element.clone().cpu().tolist()
    # indicators = data.ligand_atom_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    # indicators = None
    if basic_mode:
        indicators = None
    else:
        indicators = aromatic

    if isinstance(xyz, np.ndarray):
        xyz = xyz.tolist()
    if isinstance(atomic_nums, np.ndarray):
        atomic_nums = atomic_nums.tolist()
    if isinstance(aromatic, np.ndarray):
        aromatic = aromatic.tolist()

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, indicators)

    connect_the_dots(mol, atoms, indicators, covalent_factor=1.3)
    fixup(atoms, mol, indicators)

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, indicators)

    for (i, a) in enumerate(atoms):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, indicators)

    mol.AddHydrogens()
    fixup(atoms, mol, indicators)

    # make rings all aromatic if majority of carbons are aromatic
    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                # set all ring atoms to be aromatic
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    a.SetAromatic(True)

    # bonds must be marked aromatic for smiles to match
    for bond in ob.OBMolBondIter(mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.IsAromatic() and a2.IsAromatic():
            bond.SetAromatic(True)

    mol.PerceiveBondOrders()
    rd_mol = convert_ob_mol_to_rd_mol(mol, struct=xyz)
    try:
        # Post-processing
        rd_mol = postprocess_rd_mol_1(rd_mol)
        rd_mol = postprocess_rd_mol_2(rd_mol)
    except:
        raise MolReconError()

    return rd_mol


def reconstruct_from_generated(xyz, atomic_nums, aromatic=None, basic_mode=True):
    """
    New version of reconstruct_from_generated that handles explicit H atoms
    in generated output.

    Rules for H atoms:
    - H atoms are only generated on non-C elements (N, O, S, etc.)
    - If a non-C atom has generated H atoms, they are treated as explicit H
    - If a non-C atom has no generated H, it means the atom has no explicit H
    - C atoms never have explicit H atoms (all H on C are implicit)

    The returned RDKit molecule contains generated H atoms only -- no extra
    H atoms are added beyond what the generative model produced.

    Args:
        xyz: atom coordinates (can include H atoms)
        atomic_nums: atomic numbers (can include H=1)
        aromatic: aromatic indicators (for heavy atoms only, or all atoms)
        basic_mode: if True, ignore aromatic indicators

    Returns:
        RDKit molecule with only generated H atoms (no additional H added)
    """
    pt = AllChem.GetPeriodicTable()

    if basic_mode:
        indicators = None
    else:
        indicators = aromatic

    if isinstance(xyz, np.ndarray):
        xyz = xyz.tolist()
    if isinstance(atomic_nums, np.ndarray):
        atomic_nums = atomic_nums.tolist()
    if isinstance(aromatic, np.ndarray):
        aromatic = aromatic.tolist()

    # ==================================================================
    # Phase 1: Separate heavy atoms and H atoms
    # ==================================================================
    # Two-pass: first collect ALL heavy atoms, then assign H to nearest
    heavy_atom_indices = []   # original indices in xyz/atomic_nums
    heavy_atomic_nums = []
    heavy_xyz = []
    h_pos_list = []           # positions of all generated H atoms

    for i, (pos, anum) in enumerate(zip(xyz, atomic_nums)):
        if anum == 1:
            h_pos_list.append(pos)
        else:
            heavy_atom_indices.append(i)
            heavy_atomic_nums.append(anum)
            heavy_xyz.append(pos)

    # Assign each H to its nearest heavy atom (distance < 1.5 A)
    h_by_heavy = {}  # heavy_list_idx -> [h_pos, ...]
    if heavy_xyz and h_pos_list:
        heavy_arr = np.array(heavy_xyz)
        for h_pos in h_pos_list:
            dists = np.linalg.norm(heavy_arr - np.array(h_pos), axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < 1.5:
                h_by_heavy.setdefault(nearest, []).append(h_pos)

    # Only keep H for non-C atoms (C never has explicit generated H)
    explicit_h = {}  # heavy_list_idx -> [h_pos, ...]
    for idx, h_positions in h_by_heavy.items():
        if heavy_atomic_nums[idx] != 6:
            explicit_h[idx] = h_positions

    # Aromatic indicators mapped to heavy atoms only
    if indicators is not None:
        heavy_indicators = [indicators[i] for i in heavy_atom_indices]
    else:
        heavy_indicators = None

    # ==================================================================
    # Phase 2: Build OB molecule with heavy atoms, infer connectivity
    # ==================================================================
    ob_mol, ob_atoms = make_obmol(heavy_xyz, heavy_atomic_nums)
    fixup(ob_atoms, ob_mol, heavy_indicators)

    connect_the_dots(ob_mol, ob_atoms, heavy_indicators, covalent_factor=1.3)
    fixup(ob_atoms, ob_mol, heavy_indicators)

    ob_mol.PerceiveBondOrders()
    fixup(ob_atoms, ob_mol, heavy_indicators)

    # Aromatic ring correction (same logic as v1)
    for ring in ob.OBMolRingIter(ob_mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = 0
            aromatic_ccnt = 0
            for ai in ring._path:
                a = ob_mol.GetAtom(ai)
                if a.GetAtomicNum() == 6:
                    carbon_cnt += 1
                    if a.IsAromatic():
                        aromatic_ccnt += 1
            if carbon_cnt > 0 and aromatic_ccnt >= carbon_cnt / 2 \
                    and aromatic_ccnt != ring.Size():
                for ai in ring._path:
                    ob_mol.GetAtom(ai).SetAromatic(True)

    for bond in ob.OBMolBondIter(ob_mol):
        if bond.GetBeginAtom().IsAromatic() and bond.GetEndAtom().IsAromatic():
            bond.SetAromatic(True)

    ob_mol.PerceiveBondOrders()

    # ==================================================================
    # Phase 3: Convert OB heavy-atom mol -> RDKit mol MANUALLY
    #          (not using convert_ob_mol_to_rd_mol which adds unwanted H)
    # ==================================================================
    ob_mol.DeleteHydrogens()
    n_heavy = ob_mol.NumAtoms()

    rd_mol = AllChem.RWMol()
    rd_conf = AllChem.Conformer(n_heavy)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = AllChem.Atom(ob_atom.GetAtomicNum())
        if ob_atom.IsAromatic() and ob_atom.IsInRing() \
                and ob_atom.MemberOfRingSize() <= 6:
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        v = ob_atom.GetVector()
        rd_conf.SetAtomPosition(i, Geometry.Point3D(v.GetX(), v.GetY(), v.GetZ()))

    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i = ob_bond.GetBeginAtomIdx() - 1   # OB is 1-indexed
        j = ob_bond.GetEndAtomIdx() - 1
        order = ob_bond.GetBondOrder()
        if order == 1:
            rd_mol.AddBond(i, j, AllChem.BondType.SINGLE)
        elif order == 2:
            rd_mol.AddBond(i, j, AllChem.BondType.DOUBLE)
        elif order == 3:
            rd_mol.AddBond(i, j, AllChem.BondType.TRIPLE)
        else:
            raise Exception('unknown bond order {}'.format(order))
        if ob_bond.IsAromatic():
            rd_mol.GetBondBetweenAtoms(i, j).SetIsAromatic(True)

    # ==================================================================
    # Phase 4: Fix hypervalency on heavy-atom skeleton
    # ==================================================================
    positions = rd_mol.GetConformer().GetPositions()

    # Reduce bond order for hypervalent atoms (longest non-single first)
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() in (AllChem.BondType.DOUBLE,
                                  AllChem.BondType.TRIPLE):
            bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[bi] - positions[bj])
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for _, bond in nonsingles:
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            if bond.GetBondType() == AllChem.BondType.TRIPLE:
                bond.SetBondType(AllChem.BondType.DOUBLE)
            else:
                bond.SetBondType(AllChem.BondType.SINGLE)

    # N with 4 neighbors -> formal charge +1
    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    # C with two C=C double bonds -> demote one to single
    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetDegree() == 4:
            double_cc = []
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 6:
                    b = rd_mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
                    if b.GetBondType() == AllChem.BondType.DOUBLE:
                        double_cc.append(b)
            if len(double_cc) >= 2:
                double_cc[0].SetBondType(AllChem.BondType.SINGLE)

    # ==================================================================
    # Phase 5: Map RDKit atoms back to heavy_xyz list by position
    # ==================================================================
    rd_positions = rd_mol.GetConformer().GetPositions()
    heavy_arr = np.array(heavy_xyz)
    heavy_to_rd = {}   # heavy_list_idx -> rd_atom_idx
    rd_to_heavy = {}   # rd_atom_idx -> heavy_list_idx

    for rd_idx in range(rd_mol.GetNumAtoms()):
        dists = np.linalg.norm(heavy_arr - rd_positions[rd_idx], axis=1)
        best = int(np.argmin(dists))
        if dists[best] < 0.1 and best not in heavy_to_rd:
            heavy_to_rd[best] = rd_idx
            rd_to_heavy[rd_idx] = best

    # ==================================================================
    # Phase 6: Add generated H as explicit atoms; suppress other H
    # ==================================================================
    conf = rd_mol.GetConformer()

    # Add only the generated explicit H atoms with 3D coordinates
    for heavy_idx, h_positions in explicit_h.items():
        rd_idx = heavy_to_rd.get(heavy_idx)
        if rd_idx is None:
            continue
        for h_pos in h_positions:
            h_idx = rd_mol.AddAtom(AllChem.Atom(1))
            rd_mol.AddBond(rd_idx, h_idx, AllChem.BondType.SINGLE)
            conf.SetAtomPosition(
                h_idx, Geometry.Point3D(h_pos[0], h_pos[1], h_pos[2]))

    # For all non-C heavy atoms: suppress implicit H entirely
    # (H count is fully determined by what the model generated)
    for rd_idx in range(rd_mol.GetNumAtoms()):
        atom = rd_mol.GetAtomWithIdx(rd_idx)
        if atom.GetAtomicNum() <= 1:
            continue
        heavy_idx = rd_to_heavy.get(rd_idx)
        if heavy_idx is None:
            continue
        if heavy_atomic_nums[heavy_idx] != 6:  # non-C
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)

    # ==================================================================
    # Phase 6b: Assign formal charges for protonated atoms
    # ==================================================================
    # Now that explicit H atoms are bonded, recount each atom's actual
    # degree (heavy bonds + explicit H bonds) and set formal charges
    # where the total valence exceeds the neutral default.
    # Typical cases:
    #   N with 4 bonds (e.g. R-NH3+, quaternary N)  -> +1
    #   O with 3 bonds (e.g. R-OH2+, oxonium)       -> +1
    #   S with 3 bonds (e.g. sulfonium)              -> +1
    for rd_idx in range(rd_mol.GetNumAtoms()):
        atom = rd_mol.GetAtomWithIdx(rd_idx)
        anum = atom.GetAtomicNum()
        if anum <= 1:
            continue
        degree = atom.GetDegree()           # explicit bonds including H
        default_val = pt.GetDefaultValence(anum)
        total_bond_order = calc_valence(atom)

        if total_bond_order > default_val:
            # The extra valence is explained by a positive formal charge
            charge = int(total_bond_order - default_val)
            atom.SetFormalCharge(charge)

    # ==================================================================
    # Phase 7: Sanitize and post-process
    # ==================================================================
    rd_mol = rd_mol.GetMol()

    try:
        AllChem.SanitizeMol(rd_mol,
                            AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
    except:
        raise MolReconError()

    # Fix aromatic bond consistency (OB vs RDKit aromaticity models)
    for bond in rd_mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    # Fix radical electrons by upgrading bond orders between radical neighbors
    nbh_list = {}
    for bond in rd_mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nbh_list.setdefault(bi, []).append(bj)
        nbh_list.setdefault(bj, []).append(bi)

    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0 and idx in nbh_list:
            for j in nbh_list[idx]:
                if j <= idx:
                    continue
                nb = rd_mol.GetAtomWithIdx(j)
                if nb.GetAtomicNum() == 1:
                    continue
                nb_radical = nb.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rd_mol.GetBondBetweenAtoms(idx, j)
                    if bond.GetBondType() in UPGRADE_BOND_ORDER:
                        bond.SetBondType(
                            UPGRADE_BOND_ORDER[bond.GetBondType()])
                        nb.SetNumRadicalElectrons(nb_radical - 1)
                        num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(
                atom.GetNumExplicitHs() + num_radical)

    # Inline postprocess_rd_mol_2 logic:
    # (1) Fix bad 3-membered rings
    rdmol_edit = AllChem.RWMol(rd_mol)
    ring_info = rd_mol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rd_mol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                atom_by_symb.setdefault(symb, []).append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1)
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1)
    rd_mol = rdmol_edit.GetMol()

    # (2) Clear positive formal charges ONLY on atoms that don't need them
    #     for valence correctness. Protonated atoms (N+, O+, S+) with
    #     valence exceeding neutral default MUST keep their charge.
    for atom in rd_mol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            anum = atom.GetAtomicNum()
            default_val = pt.GetDefaultValence(anum)
            total_bond_order = calc_valence(atom)
            # Only clear charge if removing it wouldn't cause hypervalence
            if total_bond_order <= default_val:
                atom.SetFormalCharge(0)

    return rd_mol


def show_openbabel_mol(mol):
    # 遍历原子
    for atom in ob.OBMolAtomIter(mol):
        # 方法1: 原子序数
        atomic_num = atom.GetAtomicNum()

        # 方法2: 元素符号
        element_symbol = ob.GetSymbol(atomic_num)

        # 方法3: 使用GetType()获取类型
        atom_type = atom.GetType()

        print(f"原子索引: {atom.GetIdx()}")
        print(f"  原子序数: {atomic_num}")
        print(f"  元素符号: {element_symbol}")
        print(f"  原子类型: {atom_type}")
        print("-" * 30)