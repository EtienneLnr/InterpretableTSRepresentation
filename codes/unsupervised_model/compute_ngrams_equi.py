import torch
import argparse 
from sklearn.feature_extraction.text import CountVectorizer
import string
import sys 

#Custom imports
sys.path.append('../')
sys.path.append('./')

from unsupervised_model.equivariance_nn import *
from utils.train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--channel', type=int, default=64)
parser.add_argument('--n_embed', type=int, default=32)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--version', type=str, default='1')
parser.add_argument('--model_path', type=str, default='models_equi_train_k16')
parser.add_argument('--ngrams', type=int, default=2)
parser.add_argument('dataset_name', type=str)
args = parser.parse_args()


#Global path variables
save_data_path_train = '../data/' + args.dataset_name + '_TRAIN/'
save_data_path_test = '../data/' + args.dataset_name + '_TEST/'
save_model_path = '../results/trained_models'

#Load data
tensor_train = torch.load(save_data_path_train + 'X_tensor.pt').to(torch.float32)
tensor_test = torch.load(save_data_path_test + 'X_tensor.pt').to(torch.float32)
data_tensor = torch.cat([tensor_train, tensor_test], dim=0)

#Instance and load Model 
model = VQ_Equivariant_UNet(depth=args.depth, 
                            channels=args.channel, 
                            n_embed=args.n_embed, 
                            device='cpu')

model.load_state_dict(torch.load(save_model_path + '/model_' + args.dataset_name + '_' + args.version + '.pt', 
						  map_location=torch.device('cpu')))


#Forward
recons, l_p_i, l_m_p, quant, embed, vq_loss, encoding_indices = model(data_tensor)
look_up_list = list(string.ascii_letters[:args.n_embed])
text = turn_array_into_text(encoding_indices.unsqueeze(1).numpy(), look_up_list)


#Save centroides and count vectorizer
torch.save(encoding_indices, save_model_path + '/encoding_indices_' + args.dataset_name + '_' + args.version + '.pt')

vectorizer = CountVectorizer(input='content', 
                             encoding='utf-8',
                             lowercase=False, 
                             decode_error='strict', 
                             ngram_range=(1, args.ngrams),
                             analyzer='char')
                             
X_centroides = vectorizer.fit_transform(text)
X_centroides_binarized = turn_into_binary_vectors(X_centroides.toarray())
X_names = vectorizer.get_feature_names()

#save X_centroids matrix and X_names
np.save(save_model_path + '/centroides_name_' + args.dataset_name + '_' + args.version + '.npy', np.array(X_names))
np.save(save_model_path + '/centroides_' + args.dataset_name + '_' + args.version + '.npy', X_centroides_binarized)