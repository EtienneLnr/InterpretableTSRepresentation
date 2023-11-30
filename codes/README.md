
--------------------------------------------------
How to run the experiments 
--------------------------------------------------

To learn the unsupervised representations, find the best representation and then train the logistic regression on the unigrams and bigrams with the appropriate hyperparameters, you just have to run the following command: 

--> bash main.sh [dataset_name] [time_series_length]

e.g. $bash main.sh PowerCons 144  
or   $bash main.sh SmallKitchenAppliances 720  


Please refer to the data folder to check the available datasets in this folder. 
To know the length of the corresponding time series, please refer to the file DataSummary.csv.


-----------------------------
Hyperparameters
-----------------------------

The following unsupervised model hyperparameters can be changed in the main.sh file:

- The scale of reduction of the temporal dimension 
- The number of valid centroids
- The number of epochs 
- The number of channels 
- The number of epochs 
- The batch size


The following supervised model hyperparameters can be changed in the logistic_reg_equi_ensemble.py file (classifiers folder):

- The search spaces for the l1 penalty and the l2 penalty.


---------------------------------------------
Folders overview 
---------------------------------------------

----------------------------------------

- The unsupervised_model folder contains our unsupervised model, a file to train it, and a file to extract n-grams from the learned representation.

----------------------------------------

- The classifiers folder contains the content for using logistic regression over the learned representations.

----------------------------------------

- The utils folder contains a file with useful functions for training models and preprocessing tasks.

----------------------------------------


