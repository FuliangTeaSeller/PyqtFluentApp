from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import sys
sys.path.append('E:/Python/PyqtFluentApp/main/B_Pred')
from B_Pred.args import TrainArgs
from B_Pred.model import MoleculeModel
from B_Pred.data import get_data, MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
from torch.optim.lr_scheduler import ExponentialLR
import torch


def draw(smiles, node_gradient):
    prob_aro = torch.sigmoid(node_gradient).numpy().tolist()
    node_gradient = node_gradient.detach().cpu().numpy().tolist()
    nodes = []
    for pred_aro in node_gradient:
        if pred_aro >= 0:
            nodes.append(int(1))
        else:
            nodes.append(int(0))
    m = Chem.MolFromSmiles(smiles[0][0])
    rdDepictor.Compute2DCoords(m)
    drawer = rdMolDraw2D.MolDraw2DSVG(650, 650)
    drawer.SetFontSize(20)
    op = drawer.drawOptions().addAtomIndices = True
    mol = rdMolDraw2D.PrepareMolForDrawing(m)
    c = nodes
    important_index = []
    for i, value in enumerate(c):
        if value == 1:
            important_index.append(i)
    colors = [(51, 34, 136), (17, 119, 51), (68, 170, 153), (136, 204, 238), (221, 204, 119), (204, 102, 119),
              (170, 68, 153), (206, 95, 115)]
    for i, x in enumerate(colors):
        colors[i] = tuple(y / 255 for y in x)
    atom_cols = {}
    for bd in important_index:
        atom_cols[bd] = colors[i % 9]

    img = Chem.Draw.MolToImage(mol, highlightAtoms=important_index, highlightBonds=None, highlightAtomColors=atom_cols,
                               highlightAtomRadii={i: .4 for i in important_index})
    '''drawer.DrawMolecule(m, highlightAtoms=important_index, highlightBonds=None, highlightAtomColors=atom_cols,
                        highlightAtomRadii={i: .4 for i in important_index}
                        )
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().padding = .2
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    display(SVG(svg))'''
    #img.show()

    return img

#draw(model, data)