from torch.utils.data import Dataset
import torch
import numpy as np 
import os 
import pandas as pd 
import json 
import multiprocessing as mp


class MyDataset(Dataset):
    def __init__(self, TimeSeries):
        self.sequences = TimeSeries

    def __len__(self):
        return len(self.sequences[:,0, 0])

    def __getitem__(self, index):
        return self.sequences[index,:, :]


def cut_in_batch(X, nb_cut):
    
    list_of_x = list()
    list_of_nodes = list()

    for nodes_idx in range(0, nb_cut, 1):
        list_of_nodes.append(int(X.shape[0] / nb_cut) * nodes_idx)

    for nodes_idx in range(0, nb_cut, 1):

        if nodes_idx == (nb_cut -1):
            list_of_x.append(X[list_of_nodes[nodes_idx] : ])

        else:
            list_of_x.append(X[list_of_nodes[nodes_idx] :list_of_nodes[nodes_idx + 1]]) 

    return list_of_x, list_of_nodes




def pad_single_centroide(list_text_digit):

    nb_iter = len(list_text_digit)

    for ii in range(nb_iter):

        if len(list_text_digit[ii]) == 1:
            list_text_digit[ii] = '0' + list_text_digit[ii]


        else:
            pass

    return list_text_digit


def keep_track_of_ngrams(X_name):

    """
    Return a list of list with ngrams idx
    """

    list_idx_ngrams = [[] for ii in range(10)]

    for idx, ele in enumerate(X_name):

        sub_list_idx = int(len(ele) - 1)
        list_idx_ngrams[sub_list_idx].append(idx)

    return list_idx_ngrams



def turn_array_into_text(array_centroids, look_up_tables) -> list:

    list_indiv_text = list()

    for indiv_index in range(array_centroids.shape[0]):
        list_centroide_text = [look_up_tables[ele] for ele in array_centroids[indiv_index,0,:]]
        text = ''.join(list_centroide_text)
        list_indiv_text.append(text)

    return list_indiv_text



def turn_into_binary_vectors(array):

    return np.array(array > 0).astype(int)



def draw_sequences(sequence_proba):

    seq_size = sequence_proba.shape[0]
    nb_choices = sequence_proba.shape[1]

    draw_ele = list()

    for idx in range(seq_size):
        draw_ele.append(int(np.random.choice(np.arange(0, nb_choices, 1), p=sequence_proba[idx, :].numpy())))

    return torch.tensor(draw_ele)




def compute_conv1D_sequence_out_length(L_in, padding, dilat, k_s, stride):

    """
    return l_out length after conv1D
    """

    return (L_in + 2 * padding - dilat * (k_s - 1) - 1) / stride + 1




def compute_short_conv1D_sequence_out_length(L_in, k_s):

    """
    return l_out length after conv1D with padding=1 and stride=2
    """

    return (L_in + 2 - k_s ) / 2 + 1




def compute_deconv1D_sequence_out_length(L_in, padding, dilat, k_s, stride, output_padding):

    """
    return l_out length after conv1D with padding=1 and stride=2
    """

    return (L_in - 1) * stride - 2 * padding + dilat * (k_s - 1) + output_padding + 1




def compute_k_s_list_encoder(L_in, scale_reduc):

    nb_steps = int(scale_reduc)
    L_in_list = list()
    L_in_list.append(L_in)
    k_s_list = list()

    for step_idx in range(nb_steps):

        if L_in % 2 == 0:
            k_s = 4

        else:
            k_s = 3

        L_in = compute_short_conv1D_sequence_out_length(L_in, k_s)
        L_in_list.append(L_in)
        k_s_list.append(k_s)

    return L_in_list, k_s_list





def compute_decod_parameters(L_in, L_target, scale_reduc):


    L_tempo = (L_in - 1) * scale_reduc 

    if L_target - L_tempo < 3:
        padding = 1 
        k_s = L_target - (L_tempo - 2 * padding)

    else:
        padding = 0
        k_s = L_target - L_tempo

    #Check if it is right
    supposed_length = compute_deconv1D_sequence_out_length(L_in, padding, 
                                                            dilat=1, k_s=k_s, 
                                                            stride=scale_reduc, 
                                                            output_padding=0) 
    if supposed_length == L_target:
        pass
    else:
        print("Error")


    return [int(k_s)], [int(scale_reduc)], [int(padding)]



def compute_receptive_field(list_ks, list_strides):

    nb_layers = len(list_ks)
    somme = 0

    for ii in range(nb_layers):

        if ii == 0:
            prod_strides = 1
        else:
            prod_strides = np.prod(np.array(list_strides[:ii]))

        incre = (list_ks[ii] - 1) * prod_strides

        somme += incre

    return somme + 1


def compute_receptive_field_region(list_ks, list_strides, list_padding, uL, vL):

    nb_layers = len(list_ks)

    u0_part1 = uL * np.prod(np.array(list_strides))
    v0_part1 = vL * np.prod(np.array(list_strides))

    somme_u = 0 
    somme_v = 0

    for ii in range(nb_layers):

        if ii == 0:
            prod_strides = 1
        else:
            prod_strides = np.prod(np.array(list_strides[:ii]))

        incre_u = list_padding[ii] * prod_strides
        incre_v = (1 + list_padding[ii] - list_ks[ii])* prod_strides

        somme_u += incre_u
        somme_v += incre_v

    u0 = u0_part1 - somme_u
    v0 = v0_part1 - somme_v

    return u0, v0



def compute_intersection(a=[12,22], b=[13,33], compteur_a=1, compteur_b=1):

    if ((b[0] < a[1]) and (b[1] > a[0])):

        if b[0] > a[0]:
            x_inf = a[0]
            seg_inf = b[0]
            compteur_left = compteur_a
            compteur_right = compteur_b

        else:
            x_inf = b[0]
            seg_inf = a[0]
            compteur_left = compteur_b
            compteur_right = compteur_a

        if a[1] > b[1]:
            x_sup = a[1]
            seg_sup = b[1]
        else:
            x_sup = b[1]
            seg_sup = a[1]

        return [[x_inf, seg_inf], [seg_inf, seg_sup], [seg_sup, x_sup]], [compteur_left, compteur_left + compteur_right, compteur_right]

    else:

        return [a, b], [compteur_a, compteur_b]



def compute_disjoint_segment_and_weight(list_of_segment=[[12,22],[13,33],[25,67]]):

    #Init
    list_disjoint_seg_final = [list_of_segment[0]]
    list_compteurs_final = [1]

    for idx in range(1, len(list_of_segment), 1):

        list_disjoint_seg_final_copy = list()
        list_compteurs_final_copy = list()

        for idx_sub_list in range(len(list_disjoint_seg_final)):


            list_disjoint_seg, list_compteurs = compute_intersection(a=list_disjoint_seg_final[idx_sub_list], 
                                                                     b=list_of_segment[idx], 
                                                                     compteur_a=list_compteurs_final[idx_sub_list], 
                                                                     compteur_b=1)


            if len(list_disjoint_seg) != 2:

                for idx_compteur, ele in enumerate(list_disjoint_seg):

                    if ele not in list_disjoint_seg_final_copy:

                        list_disjoint_seg_final_copy.append(ele)
                        list_compteurs_final_copy.append(list_compteurs[idx_compteur])
                    else:
                        pass

            else:

                if idx_sub_list == (len(list_disjoint_seg_final) - 1):

                    for idx_compteur, ele in enumerate(list_disjoint_seg):

                        if ele not in list_disjoint_seg_final_copy:

                            list_disjoint_seg_final_copy.append(ele)
                            list_compteurs_final_copy.append(list_compteurs[idx_compteur])
                        else:
                            pass
                else: 
                    list_disjoint_seg_final_copy.append(list_disjoint_seg[0])
                    list_compteurs_final_copy.append(list_compteurs[0])

        list_disjoint_seg_final = list_disjoint_seg_final_copy
        list_compteurs_final = list_compteurs_final_copy


    return list_disjoint_seg_final, list_compteurs_final




def find_indices_for_encoding_unigrams(indice, encoding_indices):

    list_good_indices = list()

    for ii in range(len(encoding_indices)):

        if encoding_indices[ii] == indice:
            list_good_indices.append(ii)

        else:
            pass

    return list_good_indices




def find_indices_for_encoding_bigrams(indices, encoding_indices):

    list_good_indices = list()

    for ii in range(len(encoding_indices)-1):

        if (encoding_indices[ii] == indices[0]) and (encoding_indices[ii + 1] == indices[1]):
            list_good_indices.append([ii, ii+1])

        else:
            pass

    return list_good_indices


 
