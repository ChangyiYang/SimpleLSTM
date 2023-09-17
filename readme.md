# Pebble Bed with Reactor

This is the Gamma branch for Pebblebed Reactor dealing Gamma data.

## Gamma data

The Gamma data is a multi-index data. The index is time and energy channel. For each time, we collect detector's data for each energy channel. For each timestamp and each channel, the collected data is 
1D. Our preprocessing all happens for this kind of 1D data. For now we have two apporach of preprocessing, the histogram and quantile. 

## Raw Files

The related files are listed below. This is also the order of the workflow.

### Gamma_histogram.ipynb 
In this file, we use histogram approach, i.e., we count the number of data points in each bin. 


### Gamma_quantile.ipynb
In this file, we use quantile apporach, i.e., we pick several quantils to represent the whole distribution.


### Gamma_data.ipynb
In this file, we add the core data to the detector data. The core data contains keff, threshold and power. 


### Gamma_training.ipynb

In here, the model is trained and visualized. 

### Source Files

In the src folder of the repo, there are several source files. 

`TrainMethod.py ` is where the train function is defined.

`NeuralNets.py` is where the network is defined. 

`DataPrepocessing.py` is where the Pytorch dataset class is defined. 








