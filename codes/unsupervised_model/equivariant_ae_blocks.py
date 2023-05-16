import torch
import torch.nn as nn
import torch.nn.functional as F




#######################################
# E Q U I V A R I A N T.  B L O C K S
#######################################

class ApsDown(nn.Module):
    def __init__(self, channels_in, channels, kernel_size=3, stride=2, apspool_criterion = 'l2'):
        super(ApsDown, self).__init__()

        self.stride = stride  
        self.apspool_criterion = apspool_criterion
        self.conv1D = nn.Conv1d(channels_in, channels, kernel_size, stride=1, padding=1, bias=False)
        
    def forward(self, input_to_pool):
    
        inp = input_to_pool
        polyphase_indices = None
        down_func = aps_downsample_direct
        output_conv, pad = aps_pad(self.conv1D(inp))
            
        return down_func(output_conv, self.stride, polyphase_indices, apspool_criterion = self.apspool_criterion), pad


class ApsUp(nn.Module):
    def __init__(self, channels_in, channels, kernel_size=3, stride=2, apspool_criterion = 'l2', device='cpu'):
        super(ApsUp, self).__init__()
        
        self.stride = stride
        self.apspool_criterion = apspool_criterion
        self.device = device
        self.conv1D = nn.Conv1d(channels_in, channels, kernel_size, stride=1, padding=1, bias=False)
        
    def forward(self, inp, polyphase_indices, pad):

        aps_up = aps_upsample_direct(inp, self.stride, polyphase_indices, device=self.device)
        pad_oup = F.pad(aps_up, (-pad[0],-pad[1]), mode = 'constant')

        return self.conv1D(pad_oup)



def aps_downsample_direct(x, stride, polyphase_indices=None, apspool_criterion='l2'):

    if stride==1:
        return x

    elif stride>2:
        raise Exception('Stride>2 currently not supported in this implementation')

    else:
        xpoly_0 = x[:, :, ::stride]
        xpoly_1 = x[:, :, 1::stride]

        xpoly_combined = torch.stack([xpoly_0, xpoly_1], dim=1)

        if polyphase_indices is None:
            polyphase_indices = get_polyphase_indices_from_xpoly(xpoly_combined, apspool_criterion)

        B = xpoly_combined.shape[0]
        output = xpoly_combined[torch.arange(B), polyphase_indices.view(-1), :, :]
        
        return output, polyphase_indices



def get_polyphase_indices_from_xpoly(xpoly_combined, apspool_criterion):

    """
    @input : xpoly_combind, shape is (batch_size, 2*nb_channels, )
    """

    B = xpoly_combined.shape[0]

    if apspool_criterion == 'l2':
        norm_ind = 2

    elif apspool_criterion == 'l1':
        norm_ind = 1
    else:
        raise ValueError('Unknown criterion choice')

    B = xpoly_combined.shape[0]
    all_norms = torch.norm(xpoly_combined.view(B, 2, -1), dim=-1, p=norm_ind)

    return torch.argmax(all_norms, dim=1)


def aps_pad(x):

    T = x.shape[-1]
    
    if T%2==0:
        pad = (0,0)
    
    if T%2!=0:
        x = F.pad(x, (0, 1), mode = 'constant')
        pad = (0,1)
    
    return x, pad


def aps_upsample_direct(x, stride, polyphase_indices, device):
    
    if stride ==1:
        return x
    
    elif stride>2:
        raise Exception('Currently only stride 2 supported')
        
    else:
    
        B, C, T = x.shape

        y = torch.zeros(B, 2, C, T).to(x.dtype).to(device)
        y1 = torch.zeros(B, C, 2*T).to(x.dtype).to(device)
        
        y[torch.arange(B), polyphase_indices, :, :] = x

        y1[:, :, ::2] = y[:, 0, :, :]
        y1[:, :, 1::2] = y[:, 1, :, :]

        return y1



#######################################
# V E C T 0 R . Q U A N T I Z A T I O N 
#######################################

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), self._embedding.weight, perplexity, encodings, encoding_indices, distances




