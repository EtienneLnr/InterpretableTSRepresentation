import torch.optim as optim
import torch
import matplotlib.pyplot as plt 
import argparse
import sys 

#Custom imports
sys.path.append('../')
sys.path.append('./')

from unsupervised_model.equivariance_nn import *
from torch.utils.data import DataLoader
from utils.train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--channel', type=int, default=64)
parser.add_argument('--n_embed', type=int, default=10)
parser.add_argument('--num_training', type=int, default=1)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--loss_choice', type=str, default='MAE')
parser.add_argument('--version', type=str, default='1')
parser.add_argument('dataset_name', type=str)

args = parser.parse_args()

#Global path variables
save_data_path_train = '../data/' + args.dataset_name + '_TRAIN/'
save_data_path_test = '../data/' + args.dataset_name + '_TEST/'
save_model_path = '../results/trained_models'

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Data
tensor_train = torch.load(save_data_path_train + '/X_tensor.pt').to(torch.float32)
tensor_test = torch.load(save_data_path_test + '/X_tensor.pt').to(torch.float32)

#Model instance
model = VQ_Equivariant_UNet(depth=args.depth, 
                            channels=args.channel, 
                            n_embed=args.n_embed, 
                            device=device)
model = model.to(device)


#Training parameters
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
torch.cuda.empty_cache()
Myloss = nn.L1Loss() if args.loss_choice == 'MAE' else nn.MSELoss()

#Prepare for GPU's
data = MyDataset(TimeSeries=tensor_train)
training_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,pin_memory=True)

#Training
model.train()
ii = 0
list_epoch_loss = list()

while ii <= args.num_training:

    data = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    recons, l_p_i, l_m_p, quant, embed, vq_loss, encoding_indices = model(data)
    recon_error = Myloss(recons.flatten(1,2), data.flatten(1,2)) 

    loss = recon_error + vq_loss
    loss.backward()
    optimizer.step()
    ii += 1
 
    if ii%100 == 0:
        print('iter : ', ii)
        print('vq_loss', vq_loss)
        print('recons loss', recon_error)
        print('total loss', loss)
        print('unique embed', encoding_indices.unique())

    list_epoch_loss.append(recon_error.item())

torch.save(model.state_dict(), save_model_path + '/model_' + args.dataset_name + '_' + args.version + '.pt')