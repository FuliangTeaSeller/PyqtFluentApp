from dgl import DGLGraph
from typing import List, Tuple, Union
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from rdkit import Chem
import torch as th
#from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
import torch
import rdkit
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
import dgl
class Featurization_parameters:
    def __init__(self) -> None:
        self.device = 'cpu'
        self.atom_data_field = 'atom'
        self.bond_data_field = 'etype'
PARAMS = Featurization_parameters()
def find_var(df, min_value):
    df1 = df.copy()
    df1.loc["var"] = np.var(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns)if df1.iat[-1, i] < min_value]
    return col


def find_sum_0(df):
    '''
    input: df
    return: the columns of labels with no positive labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns)if df1.iat[-1, i] == 0]
    return col


def find_sum_1(df):

    '''
    input: df
    return: the columns of labels with no negative labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns)if df1.iat[-1, i] == len(df)]
    return col


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H = False, use_chirality=True):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'B',
        'C',
        'N',
        'O',
        'F',
        'Si',
        'P',
        'S',
        'Cl',
        'As',
        'Se',
        'Br',
        'Te',
        'I',
        'At',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,'other']) + [atom.GetIsAromatic()]
                # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def bond_features(bond, use_chirality=True, atompair=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats = bond_feats + one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )

    return np.array(bond_feats).astype(int)


def etype_features(bond, use_chirality=True, atompair=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    if atompair == True:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats_5 = one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
        for i, m in enumerate(bond_feats_5):
            if m == True:
                e = i
        index = index + e*64
    return index


def construct_RGCN_bigraph_from_smiles(smiles):
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    for atom_index, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom).tolist()
        atoms_feature_all.append(atom_feature)
    g.ndata["atom"] = torch.tensor(atoms_feature_all)



    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    normal_all = []
    for i in etype_feature_all:
        normal = etype_feature_all.count(i)/len(etype_feature_all)
        normal = round(normal, 1)
        normal_all.append(normal)

    g.edata["etype"] = torch.tensor(etype_feature_all)
    g.edata["normal"] = torch.tensor(normal_all)
    return g


def binary_class_split(dataset, label_name):
    zero_dataset = []
    one_dataset = []
    for i in range(len(dataset)):
        if dataset[i][label_name] == 0:
            zero_dataset.append(dataset[i])
        else:
            one_dataset.append(dataset[i])
    return zero_dataset, one_dataset


def data_set_random_split(Dataset, shuffle=False):
    if shuffle:
        np.random.shuffle(Dataset)
    train_seq    = [i for i in range(len(Dataset)) if not i % 5 == 0]
    evaluate_seq = [i for i in range(len(Dataset)) if i % 5 == 0]
    training_set = []
    val_set = []
    test_set = []
    for i in train_seq:
        training_set.append(Dataset[i])
    for i in evaluate_seq:
        if i % 2 == 0:
            val_set.append(Dataset[i])
        else:
            test_set.append(Dataset[i])
    return training_set, val_set, test_set


def get_0_1_index(df, labels):
    col_zero = [i for i, x in enumerate(df[labels]) if x == 0]
    col_one = [i for i, x in enumerate(df[labels]) if x == 1]
    return col_zero, col_one


def split_dataset(dataset):
    train_seq = [x for x in range(len(dataset)) if not x % 5 == 0]
    evaluate_seq = [x for x in range(len(dataset)) if x % 5 == 0]
    val_seq = [x for x in evaluate_seq if not x % 2 == 0]
    test_seq = [x for x in evaluate_seq if x % 2 == 0]
    train_set = dataset.loc[train_seq].reset_index(drop=True)
    val_set = dataset.loc[val_seq].reset_index(drop=True)
    test_set = dataset.loc[test_seq].reset_index(drop=True)
    return train_set, val_set, test_set


def build_dataset(dataset_smiles, label_name, smiles_name, descriptor_seq, is_descriptor=True):
    dataset_gnn = []
    labels = dataset_smiles[label_name]
    smilesList = dataset_smiles[smiles_name]
    for i, smiles in enumerate(smilesList):
        if np.isnan(labels[i]):
            continue
        try:
            g = construct_RGCN_bigraph_from_smiles(smiles)
            if is_descriptor:
                molecule = [smiles, g, [labels[i]], dataset_smiles.loc[i][descriptor_seq], build_mask(labels)]
            else:
                molecule = [smiles, g, [labels[i]], build_mask(labels)]
            dataset_gnn.append(molecule)
        except:
            print('{} is transformed failed!'.format(smiles))
    return dataset_gnn


def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i==mask_value:
            mask.append(0)
        else:
            mask.append(1)
    return mask


def multi_task_build_dataset(dataset_smiles, labels_list, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        try:
            g = construct_RGCN_bigraph_from_smiles(smiles)
            mask = build_mask(labels.loc[i], mask_value=123456)
            molecule = [smiles, g, labels.loc[i], mask, split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed!'.format(i+1, molecule_number))
        except:
            print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def built_data_and_save_for_splited(
        origin_path='example.csv',
        save_path='example.bin',
        group_path='example_group.csv',
        task_list_selected=None,
         ):
    '''
        origin_path: str
            origin csv data set path, including molecule name, smiles, task
        save_path: str
            graph out put path
        group_path: str
            group out put path
        task_list_selected: list
            a list of selected task
        '''
    data_origin = pd.read_csv(origin_path)
    smiles_name = 'smiles'
    data_origin = data_origin.fillna(123456)
    labels_list = ['p_np']#[x for x in data_origin.columns if x not in ['smiles', 'group']]
    if task_list_selected is not None:
        labels_list = task_list_selected
    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list, smiles_name=smiles_name)

    smiles, graphs, labels, mask, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels),
                    'mask': torch.tensor(mask)
                    }
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(group_path, index=None, columns=None)
    print('Molecules graph is saved!')
    save_graphs(save_path, graphs, graph_labels)


def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)


def re_standar_np(data, mean, std):
    return data*(std + 1e-10)+mean
    

def split_dataset_according_index(dataset, train_index, val_index, test_index, data_type='np'):
    if data_type =='pd':
        return pd.DataFrame(dataset[train_index]), pd.DataFrame(dataset[val_index]), pd.DataFrame(dataset[test_index])
    if data_type =='np':
        return dataset[train_index], dataset[val_index], dataset[test_index]


def load_graph_from_csv_bin_for_splited(
        bin_path='example.bin',
        group_path='example.csv'):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values
    group = pd.read_csv(group_path, index_col=None).group.to_list()
    graphs, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    '''if select_task_index is not None:
        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]'''
    # calculate not_use index
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse==0:
            not_use_index.append(index)
    train_index=[]
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index=='train' and index not in not_use_index:#training
            train_index.append(index)
        if group_index=='val' and index not in not_use_index:#valid
            val_index.append(index)
        if group_index == 'test' and index not in not_use_index:
            test_index.append(index)
    graph_List = []
    for g in graphs:
        graph_List.append(g)
    graphs_np = np.array(graphs)
    #添加extra属性
    extra = []
    Extra = []

    def describ(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        # 计算分子的属性
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        n_h_oh_count = Descriptors.NHOHCount(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        num_saturated_heterocycles = Descriptors.NumSaturatedHeterocycles(mol)
        fr_al_oh_no_tert = Descriptors.fr_Al_OH_noTert(mol)
        num_aliphatic_heterocycles = Descriptors.NumAliphaticHeterocycles(mol)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        no_count = Descriptors.NOCount(mol)
        qed = Descriptors.qed(mol)
        n_o = Descriptors.NOCount(mol)
        n_h = mol.GetNumAtoms() - num_heavy_atoms
        return [[num_heteroatoms], [num_h_donors], [n_h_oh_count], [num_h_acceptors], [num_saturated_heterocycles],
                [fr_al_oh_no_tert], [num_aliphatic_heterocycles], [no_count], [qed], [n_o], [n_h]]

    def keys(smiles):
        descriptors = []
        for j, descriptor in enumerate(Descriptors.descList):
            if j <= 199:
                mol = Chem.MolFromSmiles(smiles)
                descriptor_value = descriptor[1](mol)
                if descriptor_value > 1000000:
                    descriptors.append(round(descriptor_value / 1000, 4))
                else:
                    descriptors.append(round(descriptor_value, 4))
            else:
                break
        return descriptors
    for s in smiles:
        extra.append(describ(s))
        '''tatol_charge = 0
        mol = Chem.MolFromSmiles(s)
        mol_size = Descriptors.MolWt(mol)
        logD = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        logP = rdMolDescriptors.CalcCrippenDescriptors(mol)[1]
        #logS = rdMolDescriptors.CalcCrippenDescriptors(mol)[2]
        nHA = rdMolDescriptors.CalcNumHeteroatoms(mol)  #无键氢原子数
        #nHD = rdMolDescriptors.CalcNumHDonors(mol)  #可供氢键的受体数量
        TPSA = rdMolDescriptors.CalcTPSA(mol)  #极性表面积
        nRot = rdMolDescriptors.CalcNumRotatableBonds(mol)  #旋转键的数量
        nRing = rdMolDescriptors.CalcNumRings(mol)  #环的数量
        #MaxRing = rdMolDescriptors.CalcMaxRingSize(mol)  #最大环的大小
        nHet = rdMolDescriptors.CalcNumHeterocycles(mol) #非碳原子的数量
        fChar = rdMolDescriptors.CalcFractionCSP3(mol)  #分子的芳香性指数
        #nRig = rdMolDescriptors.CalcNumRigidBonds(mol)  #分子的刚性度量
        lipinski = rdMolDescriptors.CalcNumLipinskiHBA(mol)  # 获得分子的Lipinski氢键供体数目
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)  # 获得分子的旋转键数目
        #bertzct = BertzCT(mol)  #分子立体复杂度'''

        '''AllChem.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
        for i in charges:

            tatol_charge += i'''
        #mol_solubility = 10 ** (0.52 - 0.081 * mol_logp + 0.415 * mol_tpsa)
        '''extra.append([mol_size, logD, lipinski, num_rotatable_bonds, logP, nHA, TPSA ,nRot,
                      nRing, nHet, fChar])#, tatol_charge])'''
        #extra.append(mol_solubility)
    Extra_np = np.array(extra)
    train_smiles, val_smiles, test_smiles = split_dataset_according_index(smiles, train_index, val_index, test_index)
    train_labels, val_labels, test_labels = split_dataset_according_index(labels.numpy(), train_index, val_index,
                                                                          test_index, data_type='pd')
    train_mask, val_mask, test_mask = split_dataset_according_index(mask.numpy(), train_index, val_index, test_index,
                                                                    data_type='pd')
    train_graph, val_graph, test_graph = split_dataset_according_index(graphs_np, train_index, val_index, test_index)
    #返回Extra
    train_extra, val_extra, test_extra = split_dataset_according_index(Extra_np, train_index, val_index, test_index)
    # delete the 0_pos_label and 0_neg_label
    task_number = train_labels.values.shape[1]

    train_set = []
    val_set = []
    test_set = []
    for i in range(len(train_index)):
        molecule = [train_smiles[i], train_graph[i], train_labels.values[i], train_mask.values[i], train_extra[i]]
        train_set.append(molecule)

    for i in range(len(val_index)):
        molecule = [val_smiles[i], val_graph[i], val_labels.values[i], val_mask.values[i], val_extra[i]]
        val_set.append(molecule)

    for i in range(len(test_index)):
        molecule = [test_smiles[i], test_graph[i], test_labels.values[i], test_mask.values[i], test_extra[i]]
        test_set.append(molecule)
    print(len(train_set), len(val_set), len(test_set),  task_number)
    return train_set, val_set, test_set, task_number
class MolDGL:
    def __init__(self, mol:Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]):
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        self.g = DGLGraph()

        # Add nodes
        #mol = MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()
        self.g.add_nodes(num_atoms)
        atoms_feature_all = []
        for atom_index, atom in enumerate(mol.GetAtoms()):
            atom_feature = atom_features(atom).tolist()
            atoms_feature_all.append(atom_feature)
        self.g.ndata["atom"] = torch.tensor(atoms_feature_all)

        # Add edges
        src_list = []
        dst_list = []
        etype_feature_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            etype_feature = etype_features(bond)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
            etype_feature_all.append(etype_feature)
            etype_feature_all.append(etype_feature)

        self.g.add_edges(src_list, dst_list)
        normal_all = []
        for i in etype_feature_all:
            normal = etype_feature_all.count(i) / len(etype_feature_all)
            normal = round(normal, 1)
            normal_all.append(normal)

        self.g.edata["etype"] = torch.tensor(etype_feature_all)
        self.g.edata["normal"] = torch.tensor(normal_all)
class BatchMolDGL:
    def __init__(self, mol_dgls: List[MolDGL]):
        g_list = []
        for mol_dgl in mol_dgls:
            g_list.append(mol_dgl.g)
        self.mol_dgls = dgl.batch(g_list)
        self.mol_dgls.set_n_initializer(dgl.init.zero_initializer)
        self.mol_dgls.set_e_initializer(dgl.init.zero_initializer)
        self.atom_data_field = PARAMS.atom_data_field
        self.bond_data_field = PARAMS.bond_data_field
        self.atom_feats = None
        self.bond_feats = None
        self.device = PARAMS.device
    def get_atom_feats(self):

        self.atom_feats = self.mol_dgls.ndata.pop(self.atom_data_field).float().to(self.device)

        return self.atom_feats
    def get_bond_feats(self):

        self.bond_feats = self.mol_dgls.edata.pop(self.bond_data_field).long().to(self.device)

        return self.bond_feats


    


























