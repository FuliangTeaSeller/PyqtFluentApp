from rdkit import Chem
from rdkit.Chem import rdDepictor, MolSurf
from rdkit.Chem.Draw import rdMolDraw2D, MolToFile, _moltoimg
def predictive_toxic_group(smiles, atom_weight, label):
    atom_weight_list = atom_weight.squeeze().detach().numpy().tolist()
    max_atom_weight_index = atom_weight_list.index(max(atom_weight_list))
    significant_weight = atom_weight[max_atom_weight_index]
    mol = Chem.MolFromSmiles(smiles)
    # weight_norm = np.array(ind_weight).flatten()
    # threshold = weight_norm[np.argsort(weight_norm)[1]]
    # weight_norm = np.where(weight_norm < threshold, 0, weight_norm)
    atom_new_weight = [0 for x in range(mol.GetNumAtoms())]
    # generate most important significant circle fragment and attach significant weight
    atom = mol.GetAtomWithIdx(max_atom_weight_index)
    # # find neighbors 1
    atom_neighbors_1 = [x.GetIdx() for x in atom.GetNeighbors()]
    # find neighbors 2
    atom_neighbors_2 = []
    for neighbors_1_index in atom_neighbors_1:
        neighbor_1_atom = mol.GetAtomWithIdx(neighbors_1_index)
        atom_neighbors_2 = atom_neighbors_2 + [x.GetIdx() for x in neighbor_1_atom.GetNeighbors()]
    atom_neighbors_2.remove(max_atom_weight_index)
    # find neighbors 3
    atom_neighbors_3 = []
    for neighbors_2_index in atom_neighbors_2:
        neighbor_2_atom = mol.GetAtomWithIdx(neighbors_2_index)
        atom_neighbors_3 = atom_neighbors_3 + [x.GetIdx() for x in neighbor_2_atom.GetNeighbors()]
    atom_neighbors_3 = [x for x in atom_neighbors_3 if x not in atom_neighbors_1]
    # attach neighbor 3 significant weight
    for i in atom_neighbors_3:
        atom_new_weight[i] = significant_weight * 0.5
    for i in atom_neighbors_2:
        atom_new_weight[i] = significant_weight
    for i in atom_neighbors_1:
        atom_new_weight[i] = significant_weight
    atom_new_weight[max_atom_weight_index] = significant_weight

    significant_fg_index = [max_atom_weight_index] + atom_neighbors_1 + atom_neighbors_2 + atom_neighbors_3

    for i in range(mol.GetNumAtoms()):
        pass
        #atom_colors[i] = plt_colors.to_rgba(float(atom_new_weight[i]))
    significant_bond_index = []
    for i in range(mol.GetNumBonds()):
        Ringatom = mol.GetRingInfo().AtomRings()
        Ringbond = mol.GetRingInfo().BondRings()
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        x = atom_new_weight[u]
        y = atom_new_weight[v]
        bond_weight = (x + y) / 2
        if u in significant_fg_index and v in significant_fg_index:
            #bond_colors[i] = plt_colors.to_rgba(float(abs(bond_weight)))
            significant_bond_index.append(i)
        elif u in significant_fg_index and v in significant_fg_index and (u or v) in Ringatom:
            significant_bond_index.append()
        else:
            pass
            #bond_colors[i] = plt_colors.to_rgba(float(abs(0)))
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    img = Chem.Draw.MolToImage(mol, highlightAtoms=significant_fg_index,highlightBonds=significant_bond_index)
    return img