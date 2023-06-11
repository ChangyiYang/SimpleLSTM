# Pebble Bed Reactor with LSTM

Hi guys, welcome to Pebble Bed Reactor program. In this program, we apply LSTM nets to predict the internal state and other quantities about reactor. Breif explanations of important files are listed below. 

## Support FIles

### DataPrepocessing.py

Includes the functions for data processing. Also where the pytorch dataset is defined.

### NeuralNets.py

Where different neural network is defined. 

Actually, only one neural net is defined. We tried to add layernorm or batchnorm layer, but the existing result is not satisfying.

### TrainMethod.py

Include the train function, which will train a model with corresponding parameters. 

The grid search function is designed in the begining of this project but is never finished.

## Data Files

### Brilliantlstm.pth

The best model founded with grid search.

**Note**: the searching process is lost. Basically, the grid search is just a bunch of `for` loop.

### stable_run.csv

This file contains all the stable run data and is used for training the transiant model.

### first_run.csv and second_run.csv

There two files together are the easy run. More detailed explanation can be found in poster and pre.

### challenging_run.csv

More detailed explanation can be found in poster and pre.

## Jupter Files

### GeneralLSTM.ipynb

This file aims to trined on a LSTM based on different power and threshould for stable state and see if it works for all power and threshold.

### TransientLSTM.ipynb

This file aims to trined on a LSTM based on trainsiant data and stable data and see if it can predict transiant state well.

### TransientLSTM_analysis.ipynb

This file aims to analysis the model got from grid search. It evaluates the model's performance on the test data, which is the easy and challenging/hard run.

### isotopic_concentrations_corr.ipynb

This file aims to analysis the correlation between each isotopic. 
