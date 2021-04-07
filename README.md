# Deep Graph Convolutional Neural Network for Predicting Environmental Variables on River Networks
This repository contains code for predicting environmental variables on river networks.
The model used for these predictions is a Deep Learning model that incorporates information from the river network.
The original intent of the model was to predict stream temperature and streamflow. 

This work is being developed by researchers in the Data Science branch of the US. Geological Survey and researchers at the University of Minnesota in Vipin Kumar's lab. The orginal code was written by Xiaowei Jia.

# Running the code
There are functions for facilitating pre-processing and post-processing of the data in addition to running the model itself. We wrote a Snakemake workflow to run the entire process. 


### To run the Snakemake workflow:
    1. Install the dependencies in the `environment.yaml` file. With conda you can do this with `conda env create -f environment.yaml`
    2. Install the local `river-dl` package by `pip install path/to/river-dl/`
    3. Edit the river-dl run configuration (including paths for I/O data) in `config.yml`
    4. Run Snakemake by `snakemake --configfile config.yml`

### The data
The data used to run this model currently are specific to the Delaware River Basin but will soon be made more generic.

___

### Disclaimer
This software is in the public domain because it contains materials that originally came from the U.S. Geological Survey, an agency of the United States Department of Interior. For more information, see the official USGS copyright policy

Although this software program has been used by the U.S. Geological Survey (USGS), no warranty, expressed or implied, is made by the USGS or the U.S. Government as to the accuracy and functioning of the program and related program material nor shall the fact of distribution constitute any such warranty, and no responsibility is assumed by the USGS in connection therewith.

This software is provided “AS IS.”
