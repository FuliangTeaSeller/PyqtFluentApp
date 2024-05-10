import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List, Optional
from typing_extensions import Literal
from packaging import version
from warnings import warn

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np
from tap import Tap
#from chemprop.data import set_cache_mol, empty_cache
#from chemprop.features import get_available_features_generators
Metric = Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse']

class TrainArgs(Tap):
    in_feats: int = 40
    rgcn_hidden_feats: list = [64,64]
    classifier_hidden_feats: int = 64
    rgcn_drop_out: float = 0.2
    drop_out: float = 0.2
    task_number: int = 12
    loop: bool = True
    bin_path: str = 'E:/Python/B-Tox/data/A.bin'

    group_path: str = 'E:/Python/B-Tox/data/A.csv'

    return_weight: bool = True

    model_num: int = 1

    device: Literal['cpu', 'cuda'] = 'cpu'

    #train_class_sizes: float = 0.0

    batch_size: int = 128

    number_of_molecules: int = 1

    #num_tasks: int = 1

    num_folds: int = 1

    data_path = ('E:/Python/B-Tox/data/all.csv')#'E:/Python/chemprop-master/chemprop/data/bbbp.csv'

    classification_num: int = 1

    loss_function: Literal[
        'mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = None

    smiles_columns: List[str] = ["smiles"]

    targets_columns: List[str] = None

    split_type: Literal['random', 'scaffold_balanced'] = 'random'

    split_sizes: List[float] = [0.8,0.1,0.1]

    split_key_molecule: int = 0

    metric: Metric = None

    extra_metrics: List[Metric] = []

    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra'] = 'classification'

    log_frequency: int = 10

    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'

    """Number of folds when performing cross validation."""
    seed: int = 0

    test: bool = False
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch  randomness (e.g., random initial weights)."""
    save_dir: str = 'BBBP_3'
    """Directory where model checkpoints will be saved."""
    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    ffn_hidden_size: int = 300#=None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 30
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    no_shared_atom_bond_ffn: bool = False
    """
    Whether the FFN weights for atom and bond targets should be independent between tasks.
    """
    weights_ffn_num_layers: int = 2
    """
    Number of layers in FFN for determining weights used in constrained targets.
    """

    # Training arguments
    epochs: int = 100
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein',
                               'bounded_mse', 'bounded_mae', 'bounded_rmse'}


    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size


    @property
    def shared_atom_bond_ffn(self) -> bool:
        """
        Whether the FFN weights for atom and bond targets should be shared between tasks.
        """
        return not self.no_shared_atom_bond_ffn


    @property
    def atom_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of atomic properties.
        """
        self._atom_constraints = [False] * len(self.atom_targets)
        return self._atom_constraints

    @property
    def bond_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of bond properties.
        """
        self._bond_constraints = [False] * len(self.bond_targets)
        return self._bond_constraints

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()

        global temp_save_dir  # Prevents the temporary directory from being deleted upon function return


        # Process SMILES columns
        self.smiles_columns = ['smiles']
        '''preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )'''

        self.atom_targets, self.bond_targets = [], []


        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            temp_save_dir = TemporaryDirectory()
            self.save_dir = temp_save_dir.name


        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            elif self.dataset_type == 'multiclass':
                self.metric = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.metric = 'sid'
            elif self.dataset_type == 'regression' and self.loss_function == 'bounded_mse':
                self.metric = 'bounded_mse'
            elif self.dataset_type == 'regression':
                self.metric = 'r2'
            elif self.dataset_type == 'classification_regression':
                self.metric = 'auc'
            else:
                raise ValueError(f'Dataset type {self.dataset_type} is not supported.')


        for metric in self.metrics:
            if not any([(self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy',
                                                                              'binary_cross_entropy', 'f1', 'mcc']),
                        (self.dataset_type == 'regression' and metric in ['rmse', 'mae', 'mse', 'r2', 'bounded_rmse',
                                                                          'bounded_mae', 'bounded_mse']),
                        (self.dataset_type == 'multiclass' and metric in ['cross_entropy', 'accuracy', 'f1', 'mcc']),
                        (self.dataset_type == 'spectra' and metric in ['sid', 'wasserstein'])]):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')

        if self.loss_function is None:
            if self.dataset_type == 'classification':
                self.loss_function = 'binary_cross_entropy'
            elif self.dataset_type == 'multiclass':
                self.loss_function = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.loss_function = 'sid'
            elif self.dataset_type == 'regression':
                self.loss_function = 'mse'
            else:
                raise ValueError(f'Default loss function not configured for dataset type {self.dataset_type}.')

        if self.loss_function != 'bounded_mse' and any(
                metric in ['bounded_mse', 'bounded_rmse', 'bounded_mae'] for metric in self.metrics):
            raise ValueError(
                'Bounded metrics can only be used in conjunction with the regression loss function bounded_mse.')

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')


        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Test settings
        if self.test:
            self.epochs = 0

        # check if key molecule index is outside of the number of molecules
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError(
                'The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. ')

