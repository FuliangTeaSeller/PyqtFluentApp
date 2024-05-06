import csv
from typing import Dict, List, Set, Tuple, Union
import pandas as pd
from .feature import MolGraph
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm
from random import Random
def load_graph_for_split(data_path,smiles_name):
    data_origin = pd.read_csv(data_path)
    data_origin = data_origin.fillna(123456)
    with open(data_path) as f:
        columns = next(csv.reader(f))
        targets_columns = [target_column for target_column in columns if target_column not in [smiles_name]]

    labels = data_origin[targets_columns]
    smilesList = data_origin[smiles_name]
    data_graph = []
    for i, smiles in enumerate(smilesList):
        g = MolGraph(smiles)
        mask = build_mask(labels.loc[i], mask_value=123456)
        molecule = [smiles, g, labels.loc[i], mask]
        data_graph.append(molecule)

    train, val, test = scaffold_split(data_graph)

    return train, val , test





def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i==mask_value:
            mask.append(0)
        else:
            mask.append(1)
    return mask

def scaffold_split(data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   seed: int = 0,
                   balanced: bool = False,):
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Invalid train/val/test splits! got: {sizes}")
    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
    mols = []
    for i, d in enumerate(data):
        mol = Chem.MolFromSmiles(d[0])
        mols.append(mol)
    scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return train, val, test

def scaffold_to_smiles(mols, use_indices):

    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total = len(mols)):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = mol, includeChirality = False)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

