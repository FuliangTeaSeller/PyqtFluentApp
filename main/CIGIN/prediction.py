import warnings
import torch
from rdkit import RDLogger
from rdkit import rdBase
import sys
sys.path.append('E:/Python/pyqt5')
from .model import Cigin

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = 'cpu'

import sys, os
def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path).replace('\\','/')
def main(solute,solvent):

    device = 'cpu'
    model=Cigin().to(device)
    model.load_state_dict(torch.load(resource_path('CIGIN/cigin.tar')))
    model.eval()
    output, i_map = model(solute, solvent)

    return [output,i_map]

