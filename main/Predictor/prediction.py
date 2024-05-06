import sys
sys.path.append('E:/Python/PyqtFluentApp/main/B_Pred')
from B_Pred.data import MoleculeDataset, MoleculeDatapoint
from B_Pred.args import TrainArgs
from B_Pred.utils import load_checkpoint
import torch

dict = {
'Carcinogenicity':0.6230675578117371,
'Ames Mutagenicity':0.635084331035614,
'Respiratory toxicity':0.6175598502159119,
'Eye irritation':0.8082209825515747,
'Eye corrosion':0.7043989300727844,
'Cardiotoxicity1':0.14914588630199432,
'Cardiotoxicity10':0.6474915742874146,
'Cardiotoxicity30':0.8528580069541931,
'Cardiotoxicity5':0.4074292480945587,
'CYP1A2':0.3518486022949219,
'CYP2C19':0.2841881811618805,
'CYP2C9':0.3783678114414215,
'CYP2D6':0.18503925204277039,
'CYP3A4':0.19085778295993805,
'NR-AR':0.1093173548579216,
'NR-AR-LBD':0.009610038250684738,
'NR-AhR':0.049286000430583954,
'NR-Aromatase':0.016041185706853867,
'NR-ER':0.20859608054161072,
'NR-ER-LBD':0.04058457911014557,
'NR-PPAR-gamma':0.04711690917611122,
'SR-ARE':0.1094619631767273,
'SR-ATAD5':0.039146702736616135,
'SR-HSE':0.05485197901725769,
'SR-MMP':0.13969792425632477,
'SR-p53':0.013802326284348965
}
def main(molecule):
    data = MoleculeDataset([MoleculeDatapoint([molecule])])
    args = TrainArgs().parse_args()
    model = load_checkpoint(('E:/Python/PyqtFluentApp/main/B_Pred/MGA_model.pt'), device=args.device)
    model.eval()
    pred = model(data.batch_graph(), data.batch_dgl())
    P = torch.sigmoid(pred)
    P = P.tolist()[0]
    result = []
    for i in range(len(P)):
        if P[i] >= list(dict.values())[i]:
            result.append(1)
        else:
            result.append(0)

    return result