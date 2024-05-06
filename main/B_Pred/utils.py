import numpy as np
from logging import Logger
from typing import List, Set, Tuple, Union, Callable
import csv

import math
import os

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, Optimizer
import torch
import torch.nn as nn

from argparse import Namespace


from random import Random
from .args import TrainArgs
from .scaffold import scaffold_split
from .data import MoleculeDatapoint, MoleculeDataset, StandardScaler
from .model import MoleculeModel

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, f1_score, matthews_corrcoef
def get_class_sizes(data: MoleculeDataset, proportion: bool = True) -> List[List[float]]:
    """
    Determines the proportions of the different classes in a classification dataset.

    :param data: A classification :class:`~chemprop.data.MoleculeDataset`.
    :param proportion: Choice of whether to return proportions for class size or counts.
    :return: A list of lists of class proportions. Each inner list contains the class proportions for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError('Classification dataset must only contains 0s and 1s.')
        if proportion:
            try:
                ones = np.count_nonzero(task_targets) / len(task_targets)
            except ZeroDivisionError:
                ones = float('nan')
                print('Warning: class has no targets')
            class_sizes.append([1 - ones, ones])
        else:  # counts
            ones = np.count_nonzero(task_targets)
            class_sizes.append([len(task_targets) - ones, ones])

    return class_sizes

from scipy.stats.mstats import gmean
def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )

class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)
def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    random = Random(seed)
    if split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed)
    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def save_checkpoint(
    path: str,
    model: MoleculeModel,
    scaler: StandardScaler = None,
    args: TrainArgs = None,
) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param atom_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the atom descriptors.
    :param bond_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the bond descriptors.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
    }
    torch.save(state, path)

def load_checkpoint(
    path: str, device: torch.device = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():

        param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters

        pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    model = model.to(args.device)

    return model
def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)

def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))

def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:

    if metric == 'auc':
        return roc_auc_score
    elif metric == 'rmse':
        return rmse
    if metric == 'r2':
        return r2_score

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return mean_squared_error(targets, preds, squared=False)

def multitask_mean(
    scores: np.ndarray,
    metric: str,
    axis: int = None,
) -> float:
    """
    A function for combining the metric scores across different
    model tasks into a single score. When the metric being used
    is one that varies with the magnitude of the task (such as RMSE),
    a geometric mean is used, otherwise a more typical arithmetic mean
    is used. This prevents a task with a larger magnitude from dominating
    over one with a smaller magnitude (e.g., temperature and pressure).

    :param scores: The scores from different tasks for a single metric.
    :param metric: The metric used to generate the scores.
    :axis: The axis along which to take the mean.
    :return: The combined score across the tasks.
    """
    scale_dependent_metrics = ["rmse", "mae", "mse", "bounded_rmse", "bounded_mae", "bounded_mse"]
    nonscale_dependent_metrics = [
        "auc", "prc-auc", "r2", "accuracy", "cross_entropy",
        "binary_cross_entropy", "sid", "wasserstein", "f1", "mcc",
    ]

    if metric in scale_dependent_metrics:
        return gmean(scores, axis=axis)
    elif metric in nonscale_dependent_metrics:
        return np.mean(scores, axis=axis)
    else:
        raise NotImplementedError(
            f"The metric used, {metric}, has not been added to the list of\
                metrics that are scale-dependent or not scale-dependent.\
                This metric must be added to the appropriate list in the multitask_mean\
                function in `chemprop/utils.py` in order to be used."
        )

def get_task_names(path: str,
                   smiles_columns: Union[str, List[str]] = None
                   ) -> List[str]:

    with open(path) as f:
        header = next(csv.reader(f))

    columns = header

    ignore_columns = set(smiles_columns)

    target_names = [column for column in columns if column not in ignore_columns]

    return target_names


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.
    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns

def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header