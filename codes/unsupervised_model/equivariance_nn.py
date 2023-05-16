import torch.nn as nn
import sys 

#Custom imports
sys.path.append('../')
sys.path.append('./')
from unsupervised_model.equivariant_ae_blocks import *


class VQ_Equivariant_UNet(nn.Module):
    def __init__(self, depth, 
                 channels,
                 n_embed=32,
                 kernel_size=3, 
                 stride=2, 
                 apspool_criterion='l2', 
                 device='cpu'):
        super(VQ_Equivariant_UNet, self).__init__()

        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.activation_encoder_list = nn.ModuleList()
        self.activation_decoder_list = nn.ModuleList()
        self.encoder_res_blocks_list = nn.ModuleList()
        self.decoder_res_blocks_list = nn.ModuleList()
        self.encoder_res_blocks_activation_list = nn.ModuleList()
        self.decoder_res_blocks_activation_list = nn.ModuleList()
        self.vq = VectorQuantizerEMA(num_embeddings=n_embed, embedding_dim=channels)
        
        if depth > 1:
            for ii in range(depth):

                if ii == 0:
                    self.encoder_list.append(ApsDown(1, channels, kernel_size, stride, apspool_criterion))
                    self.decoder_list.append(ApsUp(channels, channels, kernel_size, stride, apspool_criterion, device))
                    self.activation_encoder_list.append(nn.LeakyReLU())
                    self.activation_decoder_list.append(nn.LeakyReLU())
                    self.encoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
                    self.decoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
                    self.encoder_res_blocks_activation_list.append(nn.LeakyReLU())
                    self.decoder_res_blocks_activation_list.append(nn.LeakyReLU())
                
                elif ii == (depth - 1):
                    self.encoder_list.append(ApsDown(channels, channels, kernel_size, stride, apspool_criterion))
                    self.decoder_list.append(ApsUp(channels, channels, kernel_size, stride, apspool_criterion, device))
                    self.activation_encoder_list.append(nn.LeakyReLU())
                    self.activation_decoder_list.append(nn.LeakyReLU())
                    self.encoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
                    self.decoder_res_blocks_list.append(nn.Conv1d(channels, 1, kernel_size, stride=1, padding=1, bias=False))
                    self.encoder_res_blocks_activation_list.append(nn.LeakyReLU())
                    self.decoder_res_blocks_activation_list.append(nn.Identity())            

                else:
                    self.encoder_list.append(ApsDown(channels, channels, kernel_size, stride, apspool_criterion))
                    self.decoder_list.append(ApsUp(channels, channels, kernel_size, stride, apspool_criterion, device))
                    self.activation_encoder_list.append(nn.LeakyReLU())
                    self.activation_decoder_list.append(nn.LeakyReLU())
                    self.encoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
                    self.decoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
                    self.encoder_res_blocks_activation_list.append(nn.LeakyReLU())
                    self.decoder_res_blocks_activation_list.append(nn.LeakyReLU())
        
        else:
            self.encoder_list.append(ApsDown(1, channels, kernel_size, stride, apspool_criterion))
            self.decoder_list.append(ApsUp(channels, channels, kernel_size, stride, apspool_criterion, device))
            self.activation_encoder_list.append(nn.LeakyReLU())
            self.activation_decoder_list.append(nn.LeakyReLU())
            self.encoder_res_blocks_list.append(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=1, bias=False))
            self.decoder_res_blocks_list.append(nn.Conv1d(channels, 1, kernel_size, stride=1, padding=1, bias=False))
            self.encoder_res_blocks_activation_list.append(nn.LeakyReLU())
            self.decoder_res_blocks_activation_list.append(nn.Identity()) 


    def encoder(self, x):

        list_memory_pad = []
        list_poly_indices = []
        output = x

        for idx, encoder_layers in enumerate(self.encoder_list):
            out, pad = encoder_layers(output)
            output = self.activation_encoder_list[idx](out[0])
            output = self.encoder_res_blocks_activation_list[idx](self.encoder_res_blocks_list[idx](output))
            list_memory_pad.append(pad)
            list_poly_indices.append(out[1])

        return output, list_poly_indices, list_memory_pad


    def decoder(self, x, list_poly_indices, list_memory_pad):

        output = x

        for idx, decoder_layers in enumerate(self.decoder_list):
            output = decoder_layers(output, list_poly_indices[-(idx+1)], list_memory_pad[-(idx+1)])
            output = self.activation_decoder_list[idx](output)
            output = self.decoder_res_blocks_activation_list[idx](self.decoder_res_blocks_list[idx](output))

        return output

    
    def forward(self, x):
        
        #Encoding
        output, list_poly_indices, list_memory_pad = self.encoder(x)

        #Vector quantization 
        vq_loss, quant, embed, _, _, encoding_indices, _  = self.vq(output)
        output = quant

        #Decoding
        output = self.decoder(output, list_poly_indices, list_memory_pad)


        return output, list_poly_indices, list_memory_pad, quant, embed, vq_loss, encoding_indices.view(x.shape[0],-1)

