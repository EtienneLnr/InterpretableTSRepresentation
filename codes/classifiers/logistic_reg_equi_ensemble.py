import numpy as np
import torch
import torch.nn as nn
import argparse 
from sklearn.linear_model import LogisticRegressionCV
import sklearn
import pandas as pd 
import os 
import json 
import multiprocessing
import sys

sys.path.append('../')
sys.path.append('./')

from utils.train_utils import *

#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='5')
parser.add_argument('dataset_name', type=str)
args = parser.parse_args()

#Global path variables
save_data_path_train = '../data/' + args.dataset_name + '_TRAIN/'
save_data_path_test = '../data/' + args.dataset_name + '_TEST/'
save_model_path = '../results/trained_models'
save_results_path = '../results/results_csv'

#Load X and y 
y_train = torch.load(save_data_path_train + 'y_tensor.pt').to(torch.float32).numpy()
y_test = torch.load(save_data_path_test + 'y_tensor.pt').to(torch.float32).numpy()

centroids_list = [] 
centroids_name_list = []

for v in range(0, int(args.version)):
    centroids_list.append(np.load(save_model_path + '/centroides_' + args.dataset_name + '_' + str(v) + '.npy'))
    centroids_name_list.append(np.load(save_model_path + '/centroides_name_' + args.dataset_name + '_' + str(v) + '.npy'))

list_runs_train = []
list_runs_test = []

nb_runs = 3

for r in range(0,nb_runs,1):

    #Make list
    features_space_train = []
    features_space_test = []
    list_accuracy_train = []
    list_accuracy_test = []

    #Make permutations
    permutation_train = np.random.permutation(len(y_train))
    permutation_test = np.random.permutation(len(y_test))
    y_train_shuff = y_train[permutation_train]
    y_test_shuff = y_test[permutation_test]
    centroids_train_list = [centroids[:len(y_train)][permutation_train] for centroids in centroids_list]
    centroids_test_list = [centroids[len(y_train):][permutation_test] for centroids in centroids_list]

    # Find discriminative subsequences over differents representations
        
    for idx, centroids_train in enumerate(centroids_train_list):

        list_l1_ratio = [0., 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
        list_geomspace_C = np.geomspace(1e-4, 1e6, num=100, endpoint=False)

        clf = LogisticRegressionCV(max_iter=1000000, 
                                penalty='elasticnet', 
                                multi_class='ovr',
                                cv=5, 
                                solver='saga', 
                                l1_ratios=list_l1_ratio, 
                                Cs=list_geomspace_C, n_jobs=-1).fit(centroids_train, y_train_shuff)


        learner = clf.predict(centroids_train)
        prediction = clf.predict(centroids_test_list[idx])

        Accuracy_Test = sum(abs(prediction == y_test_shuff)) / len(y_test_shuff)
        Accuracy_Train =  sum(abs(learner == y_train_shuff)) / len(y_train_shuff)

        list_accuracy_train.append(Accuracy_Train)
        list_accuracy_test.append(Accuracy_Test)

        for cl in range(clf.coef_.shape[0]):
            features_space_train.append(centroids_train.transpose()[np.argwhere(clf.coef_[cl])][:,0,:])
            features_space_test.append(centroids_test_list[idx].transpose()[np.argwhere(clf.coef_[cl])][:,0,:])


    #Concat all extracted features for different representations
    X_train = np.concatenate(features_space_train, axis=0)
    X_test = np.concatenate(features_space_test, axis=0)

    #Final regression
    list_l1_ratio = [0., 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    list_geomspace_C = np.geomspace(1e-4, 1e6, num=100, endpoint=False)

    clf = LogisticRegressionCV(max_iter=1000000,
                            penalty='elasticnet', 
                            multi_class='ovr', 
                            solver='saga',
                            cv=5, 
                            l1_ratios=list_l1_ratio, 
                            Cs=list_geomspace_C, n_jobs=-1).fit(X_train.transpose(), y_train_shuff)

    learner = clf.predict(X_train.transpose())
    prediction = clf.predict(X_test.transpose())

    final_Accuracy_Test = sum(abs(prediction == y_test_shuff)) / len(y_test_shuff)
    final_Accuracy_Train =  sum(abs(learner == y_train_shuff)) / len(y_train_shuff)

    list_accuracy_train.append(final_Accuracy_Train)
    list_accuracy_test.append(final_Accuracy_Test)
    list_runs_train.append(list_accuracy_train)
    list_runs_test.append(list_accuracy_test)


line_name = [str(ii) for ii in range(nb_runs)]
col_name = ['run_' + str(ii) for ii in range(nb_runs)]

df_results_train = pd.DataFrame(list_runs_train)
df_results_test = pd.DataFrame(list_runs_test)

df_results_train = df_results_train.transpose()
df_results_test = df_results_test.transpose()
df_results_train.columns = col_name
df_results_test.columns = col_name

df_results_train.to_csv(save_results_path + '/results_train_' + args.dataset_name + '.csv')
df_results_test.to_csv(save_results_path + '/results_test_' + args.dataset_name + '.csv')