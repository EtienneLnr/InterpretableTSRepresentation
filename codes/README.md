# Code Execution and Experiments

This README explains how to run the experiments for **Interpretable Time Series Neural Representation**.

---

## 1. How to Run Experiments

To learn the unsupervised representations, find the best representation, and then train the logistic regression on the unigrams and bigrams with the appropriate hyperparameters, simply run:

```bash
bash main.sh [dataset_name] [time_series_length]
```

e.g. `bash main.sh PowerCons 144` or `bash main.sh SmallKitchenAppliances 720`

- Please refer to the `data/` folder to check the available datasets.
- To know the corresponding time series length, please consult `DataSummary.csv`.

## 2. Hyperparameters

## 2.1 Unsupervised Model
The following hyperparameters can be modified directly in `main.sh`:
- Scale of reduction of the temporal dimension
- Number of valid centroids
- Number of epochs
- Number of channels
- Batch size
 
## 2.2 Supervised Model

The logistic regression hyperparameters can be modified in:

`codes/classifiers/logistic_reg_equi_ensemble.py`

- Search spaces for L1 penalty and L2 penalty

---

## 3. Folder Overview

### unsupervised_model

Contains:

- The unsupervised model  
- Scripts to train the model  
- Scripts to extract n-grams from the learned representation

### classifiers

Contains:

- Code to train logistic regression on the learned representations

### utils

Contains:

- Helper functions for training and preprocessing tasks

---

## 4. Notes

- Make sure your datasets are preprocessed as tensors in the `data/` folder.  
- Results and trained models will automatically be saved in the `results/` folder after running `main.sh`.
