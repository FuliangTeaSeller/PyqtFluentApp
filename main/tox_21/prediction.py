import torch
from .model import APPAT
from .molecular_graph import construct_RGCN_bigraph_from_smiles
from .predictive_toxic_group import predictive_toxic_group
device = 'cpu'
args = {}
args['in_feats'] = 40
args['rgcn_hidden_feats'] = [64, 64]
args['n_tasks'] = 12
args['return_weight'] = True
import sys, os
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")


    return os.path.join(base_path, relative_path).replace("\\",'/')

def main(molecule, module):
    g = construct_RGCN_bigraph_from_smiles(molecule)
    atom_feats = g.ndata.pop('atom').float().to(device)
    bond_feats = g.edata.pop('etype').long().to(device)
    model = APPAT(in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'], n_tasks=args['n_tasks'],
                  return_weight=args['return_weight']).to(device)
    model.load_state_dict(torch.load(resource_path('tox_21/toxicity_early_stop.pth'), map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    output, atom_weight_list, node_feats = model(g, atom_feats, bond_feats)
    labels = ['NR-AR',
              'NR-AR-LBD',
              'NR-AhR',
              'NR-Aromatase',
              'NR-ER',
              'NR-ER-LBD',
              'NR-PPAR-gamma',
              'SR-ARE',
              'SR-ATAD5',
              'SR-HSE',
              'SR-MMP',
              'SR-p53']
    #获得最终值
    P = torch.sigmoid(output)
    P = P.view(12)
    threshold = [0.8759963,0.9146815,0.7775197,0.34696525,0.6146744,0.6217907,0.8095859,0.41178977,0.7871162,0.47802973,0.61859024,0.1816867]
    result = []
    for i in range(12):
        if P[i] >= threshold[i]:
            result.append(1)
        else:
            result.append(0)
    #获取毒性基团
    imgs = []
    if module == 'single':
        for j in range(12):
            if result[j] == 1:
                atom_weight = atom_weight_list[j]
                label = labels[j]
                img = predictive_toxic_group(molecule, atom_weight, label)
                imgs.append(img)
            else:
                imgs.append([])
    else:
        pass
    return result, imgs

'''smiles = 'O=C(O)Cc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1'
main(smiles, 'single')'''
