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
import traceback

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

   
    AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)

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


def reconstruct_from_generated(xyz, atomic_nums, aromatic=None, basic_mode=True):
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
  
    # Post-processing
    rd_mol = postprocess_rd_mol_1(rd_mol)
    rd_mol = postprocess_rd_mol_2(rd_mol)
    return rd_mol


def reconstruct_from_generated_with_bonds(xyz, atomic_nums, bond_index, bond_type,
                                          aromatic=None, basic_mode=True):
    """
    Build RDKit mol from generated atoms and bonds.
    Delegates to v4: hybrid approach combining predicted bonds with distance-based
    validation, missing bond detection, and comprehensive sanitization fallbacks.
    """
    return reconstruct_from_generated_with_bonds_v2(
        xyz, atomic_nums, bond_index, bond_type,
        aromatic=aromatic, basic_mode=basic_mode,
    )


def _is_reachable_rdkit(rd_mol, start, end, exclude_bond_idx):
    """
    BFS check: can we reach 'end' from 'start' without using the bond
    identified by exclude_bond_idx?  Returns True if reachable (i.e. the
    bond is redundant and safe to remove without disconnecting the molecule).
    """
    from collections import deque
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        atom = rd_mol.GetAtomWithIdx(current)
        for bond in atom.GetBonds():
            if bond.GetIdx() == exclude_bond_idx:
                continue
            neighbor = bond.GetOtherAtomIdx(current)
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _remove_redundant_ring_bonds(rd_mol):
    """
    Remove redundant chord/diagonal bonds within rings.

    A chord bond is one whose endpoints both lie on the same SSSR ring
    but the bond itself is not an edge of that ring.  Such bonds create
    chemically invalid sub-rings and are typically prediction artifacts.

    Fused-ring shared edges, spiro atoms, and bridge bonds are correctly
    preserved because they appear as edges of their respective SSSR rings.

    Iterates until no more chords are found, removing the longest chord
    first each round (longest = most suspicious).

    Args:
        rd_mol: RDKit RWMol (modified in place)
    """
    from rdkit import Chem

    positions = rd_mol.GetConformer().GetPositions()

    changed = True
    while changed:
        changed = False
        try:
            Chem.FastFindRings(rd_mol)
        except Exception:
            break
        ring_info = rd_mol.GetRingInfo()
        rings = [list(r) for r in ring_info.AtomRings()]

        chord_candidates = []

        for ring in rings:
            if len(ring) > 6:
                continue
            ring_set = set(ring)
            # Build edge set: consecutive atom pairs in the ring
            ring_edges = set()
            
            for k in range(len(ring)):
                a, b = ring[k], ring[(k + 1) % len(ring)]
                ring_edges.add((min(a, b), max(a, b)))

            # Any bond with both endpoints in ring but NOT a ring edge is a chord
            for bond in rd_mol.GetBonds():
                ai = bond.GetBeginAtomIdx()
                bi = bond.GetEndAtomIdx()
                bond_key = (min(ai, bi), max(ai, bi))
                if ai in ring_set and bi in ring_set and bond_key not in ring_edges:
                    dist = np.linalg.norm(positions[ai] - positions[bi])
                    chord_candidates.append((dist, bond.GetIdx(), ai, bi))

        if not chord_candidates:
            break

        # Deduplicate (same bond may be chord of multiple rings)
        seen = set()
        unique = []
        for dist, bid, a, b in chord_candidates:
            if bid not in seen:
                seen.add(bid)
                unique.append((dist, bid, a, b))

        # Longest first
        unique.sort(reverse=True, key=lambda t: t[0])

        for dist, bid, a, b in unique:
            if _is_reachable_rdkit(rd_mol, a, b, exclude_bond_idx=bid):
                rd_mol.RemoveBond(a, b)
                changed = True
                break  # re-compute rings after each removal


def _infer_formal_charges(rd_mol):
    """
    Infer formal charges for non-carbon atoms based on their current valence
    and the expected default valence from the periodic table.

    Common patterns handled:
      - N with 4 bonds (e.g. quaternary amine, protonated amine) -> +1
      - N with 2 bonds in non-aromatic context -> could be -1 (deprotonated) or neutral (=NH)
      - O with 1 bond (single bond only, e.g. deprotonated carboxylate) -> -1
      - O with 3 bonds (e.g. oxonium) -> +1
      - S with valence patterns analogous to O
      - P with 4 bonds -> +1
    """
    from rdkit import Chem
    pt = AllChem.GetPeriodicTable()

    for atom in rd_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 6 or atomic_num == 1:
            # Carbon and hydrogen: no charge inference needed
            continue

        total_valence = 0.0
        for bond in atom.GetBonds():
            total_valence += bond.GetBondTypeAsDouble()
        total_valence = int(round(total_valence))

        # Get the list of allowed valences for this element
        default_valence = pt.GetDefaultValence(atomic_num)
        # RDKit returns a tuple of allowed valences or a single int
        if isinstance(default_valence, (list, tuple)):
            allowed_valences = list(default_valence)
        else:
            allowed_valences = [default_valence]

        # If the current valence matches one of the allowed valences, neutral
        if total_valence in allowed_valences:
            atom.SetFormalCharge(0)
            continue

        # Try to find a charge that makes the valence valid
        # Check charge = +1: valence should match allowed_val + 1
        # Check charge = -1: valence should match allowed_val - 1
        charge_set = False
        for charge in [1, -1, 2, -2]:
            for av in allowed_valences:
                if total_valence == av + charge:
                    atom.SetFormalCharge(charge)
                    charge_set = True
                    break
            if charge_set:
                break

        if not charge_set:
            # Heuristic fallback for specific elements
            if atomic_num == 7:  # Nitrogen
                if total_valence == 4:
                    atom.SetFormalCharge(1)
                elif total_valence == 2 and not atom.GetIsAromatic():
                    # Could be neutral (imine-like) or charged; default neutral
                    atom.SetFormalCharge(0)
            elif atomic_num == 8:  # Oxygen
                if total_valence == 1:
                    atom.SetFormalCharge(-1)
                elif total_valence == 3:
                    atom.SetFormalCharge(1)
            elif atomic_num == 16:  # Sulfur
                if total_valence == 1:
                    atom.SetFormalCharge(-1)
                elif total_valence == 3:
                    atom.SetFormalCharge(1)


def _fix_aromatic_ring_consistency(rd_mol):
    """
    Ensure aromatic ring consistency: if a ring has most atoms marked aromatic,
    mark all ring atoms and bonds as aromatic. Also fix partially aromatic rings.
    """
    from rdkit import Chem
    # Ensure ring info is computed
    try:
        Chem.FastFindRings(rd_mol)
    except Exception:
        pass
    ring_info = rd_mol.GetRingInfo()
    # if not ring_info.isInitialized():
    #     return
    for ring in ring_info.AtomRings():
        if len(ring) < 5 or len(ring) > 6:
            continue
        aromatic_count = sum(1 for idx in ring if rd_mol.GetAtomWithIdx(idx).GetIsAromatic())
        if aromatic_count >= len(ring) // 2 + 1:
            # Mark all ring atoms as aromatic
            for idx in ring:
                rd_mol.GetAtomWithIdx(idx).SetIsAromatic(True)
            # Mark all ring bonds as aromatic
            for k in range(len(ring)):
                bond = rd_mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % len(ring)])
                if bond is not None:
                    bond.SetIsAromatic(True)
                    bond.SetBondType(AllChem.BondType.AROMATIC)
                    # bond.SetBondType(AllChem.BondType.SINGLE)



def _fix_nonring_aromatic(rd_mol):
    """
    Clear aromatic flags from atoms and bonds that are not in any ring.
    RDKit requires aromatic atoms/bonds to be part of a ring; non-ring
    aromatic flags are prediction artifacts that cause sanitization failures.
    Aromatic bonds on non-ring atoms are downgraded to SINGLE.
    """
    from rdkit import Chem
    try:
        Chem.FastFindRings(rd_mol)
    except Exception:
        pass

    # Fix non-ring aromatic atoms
    for atom in rd_mol.GetAtoms():
        if atom.GetIsAromatic() and not atom.IsInRing():
            atom.SetIsAromatic(False)

    # Fix non-ring aromatic bonds (or bonds where either endpoint lost aromaticity)
    for bond in rd_mol.GetBonds():
        if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if not bond.IsInRing() or not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
                bond.SetBondType(Chem.BondType.SINGLE)


def _fix_hypervalent_bonds(rd_mol):
    """
    Fix hypervalent atoms by downgrading bond orders (longest non-single bonds first),
    taking into account explicit H neighbors.
    """
    pt = AllChem.GetPeriodicTable()
    positions = rd_mol.GetConformer().GetPositions()

    # Collect non-single bonds sorted by distance (longest first = most suspicious)
    nonsingles = []
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() in (AllChem.BondType.DOUBLE, AllChem.BondType.TRIPLE):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(positions[i] - positions[j])
            nonsingles.append((dist, bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for (d, bond) in nonsingles:
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        v1 = calc_valence(a1)
        v2 = calc_valence(a2)

        # Get max valence considering formal charge
        max1 = pt.GetDefaultValence(a1.GetAtomicNum())
        max2 = pt.GetDefaultValence(a2.GetAtomicNum())
        if isinstance(max1, (list, tuple)):
            max1 = max(max1)
        if isinstance(max2, (list, tuple)):
            max2 = max(max2)

        if v1 > max1 + a1.GetFormalCharge() or v2 > max2 + a2.GetFormalCharge():
            if bond.GetBondType() == AllChem.BondType.TRIPLE:
                bond.SetBondType(AllChem.BondType.DOUBLE)
            else:
                bond.SetBondType(AllChem.BondType.SINGLE)


def reconstruct_from_generated_with_bonds_v2(xyz, atomic_nums, bond_index, bond_type,
                                              aromatic=None, basic_mode=True):
    """
    Build RDKit mol from generated atoms (including explicit non-C hydrogens) and bonds.

    Key improvements over v1:
      1. Non-C hydrogen atoms (present in the input) are treated as explicit atoms
         in the molecule, participating in bonding and valence calculations.
      2. Formal charges are inferred for non-carbon atoms based on their observed
         valence vs. expected valence (handles protonation, deprotonation, etc.).
      3. Valence/bond-order fixing accounts for explicit H neighbors.
      4. The output molecule retains all generated H atoms with their 3D coordinates.

    Args:
        xyz: [N, 3] coordinates (numpy array or list)
        atomic_nums: [N] atomic numbers (list of int), including H (1) for non-C hydrogens
        bond_index: [2, E] atom pair indices for predicted bonds
        bond_type: [E] bond types (0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic)
        aromatic: optional per-atom aromatic flags, kept for API compat
        basic_mode: kept for API compat
    Returns:
        RDKit Mol object with explicit non-C hydrogens and inferred charges
    Fallback:
        If sanitization fails, falls back to reconstruct_from_generated
    """
    from rdkit import Chem

    if isinstance(xyz, np.ndarray):
        xyz = xyz.tolist()
    if isinstance(atomic_nums, np.ndarray):
        atomic_nums = atomic_nums.tolist()

    n_atoms = len(atomic_nums)
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # Track which atoms are H bonded to C (to distinguish from non-C H)
    h_indices = set()

    # Step 1: Add all atoms (including generated H)
    for i, atom_num in enumerate(atomic_nums):
        rd_atom = Chem.Atom(int(atom_num))
        # Mark H atoms; we'll determine C-H vs non-C-H after bonds are added
        if int(atom_num) == 1:
            h_indices.add(i)
        # Don't set implicit H count yet — let the pipeline handle it
        rd_atom.SetNoImplicit(False)
        rd_mol.AddAtom(rd_atom)
        rd_conf.SetAtomPosition(i, Geometry.Point3D(*xyz[i]))
    rd_mol.AddConformer(rd_conf)

    # Step 2: Add bonds from generated bond predictions
    BOND_MAP = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
        # 4: Chem.BondType.SINGLE
    }

    added_bonds = set()
    for idx in range(bond_index.shape[1]):
        i_atom = int(bond_index[0, idx])
        j_atom = int(bond_index[1, idx])
        bt = int(bond_type[idx])
        if bt not in BOND_MAP:
            continue
        # Ensure each bond is only added once (handle both i<j and i>j)
        bond_key = (min(i_atom, j_atom), max(i_atom, j_atom))
        if bond_key in added_bonds:
            continue
        added_bonds.add(bond_key)

        rd_mol.AddBond(i_atom, j_atom, BOND_MAP[bt])
        if bt == 4:
            rd_mol.GetAtomWithIdx(i_atom).SetIsAromatic(True)
            rd_mol.GetAtomWithIdx(j_atom).SetIsAromatic(True)
            bond_obj = rd_mol.GetBondBetweenAtoms(i_atom, j_atom)
            if bond_obj is not None:
                bond_obj.SetIsAromatic(True)

    # Step 2.5: Remove redundant chord bonds within rings
    _remove_redundant_ring_bonds(rd_mol)

    # Step 3: For each H atom, determine if it's bonded to C or non-C
    # Non-C H atoms are the key ones we want to keep explicit
    c_h_indices = set()
    non_c_h_indices = set()
    for h_idx in h_indices:
        atom = rd_mol.GetAtomWithIdx(h_idx)
        bonded_to_carbon = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                bonded_to_carbon = True
                break
        if bonded_to_carbon:
            c_h_indices.add(h_idx)
        else:
            non_c_h_indices.add(h_idx)

    # Step 4: Mark all generated H atoms as explicit (set NoImplicit on the H atoms)
    # This ensures RDKit doesn't try to remove or re-add them
    for h_idx in h_indices:
        atom = rd_mol.GetAtomWithIdx(h_idx)
        atom.SetNoImplicit(True)

    # Step 5: For heavy atoms bonded to generated H, reduce their implicit H count
    # so the total H count is correct
    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        # Count how many explicit H neighbors this heavy atom has from generated data
        explicit_h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
        if explicit_h_count > 0:
            # Let RDKit figure out remaining implicit H after charge inference
            atom.SetNoImplicit(False)

    # Step 6: Fix hypervalent atoms (considering explicit H in valence)
    _fix_hypervalent_bonds(rd_mol)

    # Step 7: Fix aromatic ring consistency
    _fix_aromatic_ring_consistency(rd_mol)

    # Step 7.5: Clear aromatic flags from non-ring atoms/bonds
    _fix_nonring_aromatic(rd_mol)

    # Step 8: Infer formal charges for non-C atoms
    _infer_formal_charges(rd_mol)


    # Step 9: Fix remaining aromatic bond/atom consistency
    # for bond in rd_mol.GetBonds():
    #     a1 = bond.GetBeginAtom()
    #     a2 = bond.GetEndAtom()
    #     if bond.GetIsAromatic():
    #         if not a1.GetIsAromatic() or not a2.GetIsAromatic():
    #             bond.SetIsAromatic(False)
    #             bond.SetBondType(Chem.BondType.SINGLE)
    #     elif a1.GetIsAromatic() and a2.GetIsAromatic():
    #         bond.SetIsAromatic(True)
    #         bond.SetBondType(Chem.BondType.AROMATIC)

    # Step 10: Try sanitization
    mol = rd_mol.GetMol()
    before_smiles = Chem.MolToSmiles(mol)
    after_sanitize_smiles = ""
    kekulize_exp="no"
    after_kekulize_smiles = ""
    try:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        # Try kekulization separately — if it fails, still keep the mol
        after_sanitize_smiles = Chem.MolToSmiles(mol)
      
        try:
            Chem.Kekulize(mol, clearAromaticFlags=False)
            after_kekulize_smiles = Chem.MolToSmiles(mol)
        except Exception:
            kekulize_exp="yes"
            pass
        # print(f"Before: {before_smiles},\nAfter Sanitize: {after_sanitize_smiles},\nAfter Kekulize: {after_kekulize_smiles}, keklize_exp: {kekulize_exp}")

    except Exception as e:
        print(f"[reconstruct_v2] SanitizeMol failed: {type(e).__name__}: {e}")
        # Try a more lenient sanitization: skip properties that commonly fail
        try:
            mol = rd_mol.GetMol()
            Chem.SanitizeMol(mol, Chem.SANITIZE_FINDRADICALS | Chem.SANITIZE_SETAROMATICITY |
                             Chem.SANITIZE_SETCONJUGATION | Chem.SANITIZE_SETHYBRIDIZATION |
                             Chem.SANITIZE_SYMMRINGS)
        except Exception as e2:
            print(f"[reconstruct_v2] Lenient sanitization also failed: {type(e2).__name__}: {e2}")
            # Final fallback to heuristic reconstruction
            try:
                mol = reconstruct_from_generated(xyz, atomic_nums, aromatic=aromatic, basic_mode=basic_mode)
                print(f"[reconstruct_v2] Fallback SMILES: {Chem.MolToSmiles(mol)}")
                return mol
            except Exception as e3:
                print(f"[reconstruct_v2] Fallback also failed: {type(e3).__name__}: {e3}")
                return rd_mol.GetMol()
    return mol

def reconstruct_from_generated_with_bonds_v3(xyz, atomic_nums, bond_index, bond_type, 
                                             aromatic=None, basic_mode=True):
    """
    优化版：增强了对芳香键和价键错误的鲁棒性处理。
    """
    from rdkit import Chem
    
    # 基础数据转换
    if isinstance(xyz, np.ndarray): xyz = xyz.tolist()
    if isinstance(atomic_nums, np.ndarray): atomic_nums = atomic_nums.tolist()

    n_atoms = len(atomic_nums)
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # Step 1: 添加原子
    for i, atom_num in enumerate(atomic_nums):
        rd_atom = Chem.Atom(int(atom_num))
        # 默认关闭隐式氢处理，由我们后续精细控制
        rd_atom.SetNoImplicit(True) 
        rd_mol.AddAtom(rd_atom)
        rd_conf.SetAtomPosition(i, Geometry.Point3D(*xyz[i]))
    rd_mol.AddConformer(rd_conf)

    # Step 2: 添加键
    BOND_MAP = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }

    added_bonds = set()
    for idx in range(bond_index.shape[1]):
        i_atom = int(bond_index[0, idx])
        j_atom = int(bond_index[1, idx])
        bt = int(bond_type[idx])
        if bt not in BOND_MAP:
            continue
        # Ensure each bond is only added once (handle both i<j and i>j)
        bond_key = (min(i_atom, j_atom), max(i_atom, j_atom))
        if bond_key in added_bonds:
            continue
        added_bonds.add(bond_key)

        rd_mol.AddBond(i_atom, j_atom, BOND_MAP[bt])
        if bt == 4:
            rd_mol.GetAtomWithIdx(i_atom).SetIsAromatic(True)
            rd_mol.GetAtomWithIdx(j_atom).SetIsAromatic(True)
            bond_obj = rd_mol.GetBondBetweenAtoms(i_atom, j_atom)
            if bond_obj is not None:
                bond_obj.SetIsAromatic(True)
                
    # --- 核心优化区：预清洗与修复 ---
    
    # A. 修复非环芳香键 (Aromatic bonds must be in a ring)
    # RDKit 中芳香键必须存在于环内。如果模型预测了链状芳香键，直接降级为单键。
    ring_info = rd_mol.GetRingInfo()
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            if not bond.IsInRing():
                bond.SetBondType(Chem.BondType.SINGLE)
                bond.SetIsAromatic(False)

    # B. 调用你原有的修复函数（确保它们能处理 RWMol）
    try:
        if '_fix_hypervalent_bonds' in globals():
            _fix_hypervalent_bonds(rd_mol)
        if '_infer_formal_charges' in globals():
            _infer_formal_charges(rd_mol)
    except Exception as e:
        print(f"Pre-fix logic failed: {e}")

    # C. 芳香性一致性强制修正
    # 如果一个原子被标记为芳香，但它连接的所有键都不是芳香键，则清除该原子的芳香标志
    for atom in rd_mol.GetAtoms():
        if atom.GetIsAromatic():
            has_aromatic_bond = any(b.GetIsAromatic() for b in atom.GetBonds())
            if not has_aromatic_bond:
                atom.SetIsAromatic(False)

    # Step 10: 分阶段 Sanitization
    mol = rd_mol.GetMol()
    
    try:
        # 第一步：尝试除 Kekulize 之外的所有检查
        # 这样即使芳香电子数不对，分子对象依然能生成
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception as e:
        print(f"Initial sanitize failed, trying minimal: {e}")
        # 极简模式：只保证基础环和价键信息
        params = Chem.SanitizeFlags.SANITIZE_FINDRADICALS | \
                 Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | \
                 Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        try:
            Chem.SanitizeMol(mol, params)
        except:
            pass # 最后的倔强，交给 fallback

    # 尝试 Kekulize
    try:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception:
        # 如果 Kekulize 失败，通常是由于“伪芳香环”
        # 策略：将所有 AROMATIC 键降级为 SINGLE/DOUBLE 尝试强制修复
        try:
            for b in mol.GetBonds():
                if b.GetBondType() == Chem.BondType.AROMATIC:
                    b.SetBondType(Chem.BondType.SINGLE)
                    b.SetIsAromatic(False)
            for a in mol.GetAtoms():
                a.SetIsAromatic(False)
            Chem.SanitizeMol(mol)
        except:
            # 最终 fallback 到你原有的启发式重建
            return reconstruct_from_generated(xyz, atomic_nums, aromatic=aromatic, basic_mode=basic_mode)

    return mol

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
