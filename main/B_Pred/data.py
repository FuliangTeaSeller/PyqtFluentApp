import csv

import dgl
from tqdm import tqdm
from logging import Logger
import os
import logging
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from typing import Dict, Iterator, List, Optional, Union, Tuple, Any
from torch.utils.data import DataLoader, Dataset, Sampler
from random import Random
from rdkit import Chem
from .feature import BatchMolGraph,MolGraph
from .DGL_feature import MolDGL, BatchMolDGL

from collections import OrderedDict,defaultdict

class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

SMILES_TO_GRAPH: Dict[str, MolGraph] = {}
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}
def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path: str, smiles_columns: Union[str, List[str]] = None,) -> List[str]:
    columns = get_header(path)
    smiles_columns = smiles_columns
    ignore_columns = set(smiles_columns)
    target_names = [column for column in columns if column not in ignore_columns]
    return target_names
def get_data(path: str,args):

    smiles_columns = args.smiles_columns
    target_columns =args.targets_columns
    loss_function = args.loss_function
    features_data = None
    phase_features = None
    constraints_data = None
    raw_constraints_data = None
    data_weights = None
    if target_columns is None:
        target_columns = get_task_names(
            path=path,
            smiles_columns=smiles_columns,
        )

    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_smiles, all_targets = [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets = []
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                else:
                    targets.append(float(value))
            all_smiles.append(smiles)
            all_targets.append(targets)
        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                               total=len(all_smiles))
        ])
    return data
def make_mols(smiles: List[str]):
    mol = []
    for s in smiles:
        mol.append(make_mol(s))
    return mol


def make_mol(s: str, keep_h: bool=False, add_h: bool=False, keep_atom_map: bool=False):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map and mol is not None:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)
    elif not keep_atom_map and mol is not None:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    return mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 features: np.ndarray = None,
                 data_weight: float = None,
                 ):
        self.smiles = smiles
        self.targets = targets
        self.features = features

        if data_weight is not None:
            self.data_weight = data_weight

        self.raw_features, self.raw_targets = self.features, self.targets


    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = make_mols(self.smiles)
        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [len(self.mol[i].GetAtoms()) for i in range(self.number_of_molecules)]

    @property
    def number_of_bonds(self) -> List[int]:
        """
        Gets the number of bonds in the :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [len(self.mol[i].GetBonds()) for i in range(self.number_of_molecules)]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in the :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [[b.GetBondTypeAsDouble() for b in self.mol[i].GetBonds()] for i in range(self.number_of_molecules)]

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets
class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._batch_dgl = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]
    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    @property
    def number_of_atoms(self) -> List[List[int]]:
        """
        Gets the number of atoms in each :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [d.number_of_atoms for d in self._data]

    @property
    def number_of_bonds(self) -> List[List[int]]:
        """
        Gets the number of bonds in each :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [d.number_of_bonds for d in self._data]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in each :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [d.bond_types for d in self._data]
    def batch_dgl(self) -> List[MolDGL]:
        if self._batch_dgl is None:
            self._batch_dgl = []

            mol_dgls = []
            for d in self._data:
                mol_dgls_list = []
                for s, m in zip(d.smiles, d.mol):
                    if len(d.smiles) > 1 :
                        raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                    'per input (i.e., number_of_molecules = 1).')
                    mol_dgl = MolDGL(s)
                    mol_dgls_list.append(mol_dgl)
                mol_dgls.append(mol_dgls_list)
            self._batch_dgl = [BatchMolDGL([g[i] for g in mol_dgls]) for i in range(len(mol_dgls[0]))]
        return self._batch_dgl

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if len(d.smiles) > 1 :
                        raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                    'per input (i.e., number_of_molecules = 1).')

                    mol_graph = MolGraph(m)
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph
    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None
    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]

    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each molecule and task are present.

        :return: A list of list of booleans associated with targets.
        """
        targets = self.targets()
        mask = [[t is not None for t in dt] for dt in targets]
        mask = list(zip(*mask))

        return mask

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]
def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()# Forces computation and caching of the BatchMolGraph for the molecules
    data.batch_dgl()
    return data
class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length
class MoleculeDataLoader(DataLoader):

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()



def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Split sizes do not sum to 1. Received train/val/test splits: {sizes}")
    if any([size < 0 for size in sizes]):
        raise ValueError(f"Split sizes must be non-negative. Received train/val/test splits: {sizes}")

    random = Random(seed)

    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed, logger=logger)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')




