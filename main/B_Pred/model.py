from typing import List, Union, Tuple
from DGL_feature import MolDGL, BatchMolDGL
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
#from .mpn import MPN
from args import TrainArgs
from feature import BatchMolGraph, get_bond_fdim, get_atom_fdim
from nn_utils import initialize_weights, get_activation_function
from DGL_model import MGA
from dgl.nn.pytorch.conv import RelGraphConv,GATConv,APPNPConv
from dgl.readout import sum_nodes
class MoleculeModel(nn.Module):

    def __init__(self, args: TrainArgs):
        super(MoleculeModel, self).__init__()

        self.return_weight = args.return_weight

        self.classification = args.dataset_type == "classification"
        self.loss_function = args.loss_function

        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        self.relative_output_size = 1

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.create_encoder(args)
        self.create_MGA(args)
        self.create_ffn(args)

        self.Linner = nn.Linear(2*args.num_tasks, args.num_tasks)
        #initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        self.encoder = MPN(args)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def create_MGA(self, args: TrainArgs)-> None:
        self.MGA = MGA(in_feats=args.in_feats, rgcn_hidden_feats=args.rgcn_hidden_feats, n_tasks=args.num_tasks,
                       return_weight=self.return_weight,
                       rgcn_drop_out=args.rgcn_drop_out,classifier_hidden_feats=args.classifier_hidden_feats,
                       dropout=args.drop_out, loop=args.loop)

    def create_ffn(self, args: TrainArgs) -> None:
        first_linear_dim = args.hidden_size * args.number_of_molecules
        atom_first_linear_dim = first_linear_dim
        bond_first_linear_dim = first_linear_dim
        self.readout = build_ffn(
            first_linear_dim=atom_first_linear_dim,
            hidden_size=args.ffn_hidden_size,
            num_layers=args.ffn_num_layers,
            output_size=self.relative_output_size * args.num_tasks,
            dropout=args.dropout,
            activation=args.activation
        )
    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
    ) -> torch.Tensor:
        return self.encoder(
            batch,
            features_batch
        )

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        dgls: Union[
            List[BatchMolDGL],
            List[MolDGL]
        ]
    ) -> Union[
        torch.Tensor,
        tuple,
    ]:
        if self.return_weight:
            encodings = self.encoder(
                batch
            )
            output = self.readout(encodings)
            if self.classification:
                output = self.sigmoid(output)
            for dgl in dgls:
                atom_feats = dgl.get_atom_feats()
                bond_feats = dgl.get_bond_feats()
            MGA_output, node_gradient = self.MGA(dgls, atom_feats, bond_feats)
            #MGA_ouput = self.sigmoid(MGA_ouput)
            output = self.Linner(torch.cat([output, MGA_output], dim=1))
            return output, MGA_output, node_gradient
        else:
            encodings = self.encoder(
                batch
            )
            output = self.readout(encodings)
            if self.classification:
                output = self.sigmoid(output)
            for dgl in dgls:
                atom_feats = dgl.get_atom_feats()
                bond_feats = dgl.get_bond_feats()
            MGA_output = self.MGA(dgls, atom_feats, bond_feats)
            # MGA_ouput = self.sigmoid(MGA_ouput)
            output = self.Linner(torch.cat([output, MGA_output], dim=1))

        return output

class MPN(nn.Module):
    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):

        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim()
        self.device = args.device
        self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                        for _ in range(args.number_of_molecules)])


    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[
                BatchMolGraph]],
                ) -> torch.Tensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = encodings[0] if len(encodings) == 1 else torch.cat(encodings, dim=1)

        return output
class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension.
        :param bias: Whether to add bias to linear layers.
        :param depth: Number of message passing steps.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        #self.atom_messages = args.atom_messages
        self.hidden_size = hidden_size or args.hidden_size
        self.bias = bias or args.bias
        self.depth = depth or args.depth
        self.layers_per_message = 1
        #self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        #self.is_atom_bond_targets = args.is_atom_bond_targets

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.bond_fdim #self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)


        w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)



    def forward(self,
                mol_graph: BatchMolGraph,
                ) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x hidden

        # atom hidden
        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden


        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden

def build_ffn(
    first_linear_dim: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float,
    activation: str,
) -> nn.Sequential:
    """
    Returns an `nn.Sequential` object of FFN layers.

    :param first_linear_dim: Dimensionality of fisrt layer.
    :param hidden_size: Dimensionality of hidden layers.
    :param num_layers: Number of layers in FFN.
    :param output_size: The size of output.
    :param dropout: Dropout probability.
    :param activation: Activation function.
    :param dataset_type: Type of dataset.
    :param spectra_activation: Activation function used in dataset_type spectra training to constrain outputs to be positive.
    """
    activation = get_activation_function(activation)

    if num_layers == 1:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, output_size)
        ]
    else:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, hidden_size)
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ])
        layers.extend([
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        ])

    return nn.Sequential(*layers)

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target

def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = []
    for i, g in enumerate(graphs):
        graph_list = []
        g.normalize_features(replace_nan_token=0)
        graph_list.append(g)
    bg.append(graph_list)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)

    return smiles, bg, labels, mask


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        feat_list = []
        atom_list = []
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )
class BaseGNN(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks, rgcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gnn_out_feats#128+64
        self.return_mol_embedding=return_mol_embedding

        #self.fc_extra = nn.ModuleList([self.fc_extra_layer(200, 128) for _ in range(self.task_num)])
        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])
        #self.extra_bn = nn.BatchNorm1d(200)
        #self.softmax = nn.Softmax(dim=1)
    def forward(self,
                mol_dgl: BatchMolDGL,
                node_feats, etype, norm=None): # bg
        # Update atom features with GNNs
        bg = mol_dgl[0].mol_dgls
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype)

        # Compute molecule features from atom features
        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            #mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            mol_feats = feats_list[i]
            #extra = extra.squeeze(dim=-1)
            #Extra = self.fc_extra[i](extra)
            #mol_feats = torch.cat([feats_list[i], Extra], dim=1)
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight and atom feats
            if self.return_weight:
                node_gradient_all = []
                for i in range(self.task_num):
                    device = 'cpu'
                    baseline = torch.zeros(node_feats.shape).to(device)
                    scaled_nodefeats = [baseline + (float(i) / 50) * (node_feats - baseline) for i in range(0, 51)]
                    gradients = []
                    for scaled_nodefeat in scaled_nodefeats:
                        scaled_hg, scaled_atom_weight = self.weighted_sum_readout(bg, scaled_nodefeat)
                        scaled_feats = scaled_hg[i]
                        h1 = self.fc_layers1[i](scaled_feats)
                        h2 = self.fc_layers2[i](h1)
                        h3 = self.fc_layers3[i](h2)
                        scaled_Final_feature = self.output_layer1[i](h3)
                        gradient = torch.autograd.grad(scaled_Final_feature[0][0], scaled_nodefeat)[0]
                        gradient = gradient.detach().cpu().numpy()
                        gradients.append(gradient)
                    gradients = np.array(gradients)
                    grads = (gradients[:-1] + gradients[1:]) / 2.0
                    avg_grads = np.average(grads, axis=0)
                    avg_grads = torch.from_numpy(avg_grads).to(device)
                    integrated_gradients = (node_feats - baseline) * avg_grads
                    phi0 = []
                    for j in range(node_feats.shape[0]):
                        a = sum(integrated_gradients[j].detach().cpu().numpy().tolist())
                        phi0.append(a)
                    node_gradient = torch.tensor(phi0)
                    node_gradient_all.append(node_gradient)
                return prediction_all, node_gradient_all#atom_weight_list, node_feats
            # just generate prediction
            return prediction_all


    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )
    def fc_extra_layer(self, in_feats, out_feats):
        return nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.Sigmoid(),
            nn.BatchNorm1d(out_feats)
        )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, out_feats)
                )

class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        '''self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=None, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)'''
        self.graph_conv_layer = GATConv(in_feats, out_feats, num_heads=4, feat_drop=rgcn_drop_out,
                                        attn_drop=0.2, residual=residual, activation=activation)
        self.appnp = APPNPConv(k=5, alpha=0.5)
        self.convd = nn.Conv1d(4, 1, kernel_size=1)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self,
                mol_dgl: BatchMolDGL,
                node_feats, etype,): #bg,  norm=None
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        bg = mol_dgl
        '''node_feats = mol_dgl.get_atom_feats()
        etype = mol_dgl.get_bond_feats()'''
        new_feats = self.graph_conv_layer(bg, node_feats)# etype, norm)
        res_feats = self.activation(self.res_connection(node_feats))
        new_feats = self.convd(new_feats)
        new_feats = new_feats.view(new_feats.size(0), -1)
        if self.residual:
            #new_feats = self.appnp(bg, new_feats)
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats

class MGA(BaseGNN):
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.):
        super(MGA, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                  n_tasks=n_tasks,
                                  classifier_hidden_feats=classifier_hidden_feats,
                                  return_mol_embedding=return_mol_embedding,
                                  return_weight=return_weight,
                                  rgcn_drop_out=rgcn_drop_out,
                                  dropout=dropout,
                                  )

        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats
