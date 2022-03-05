# Deep Graph Convolutional Neural Network for Predicting Environmental Variables on River Networks
This repository contains code for predicting environmental variables on river networks.  The models included are all either
temporally or spatiotemporally aware and incorporate information from the river network. The original intent of 
this repository was to predict stream temperature and streamflow. 

This work is being developed by researchers in the Data Science branch of the US. Geological Survey and researchers at the 
University of Minnesota in Vipin Kumar's lab. Sources for specific models are included as comments within the code.

# Running the code
There are functions for facilitating pre-processing and post-processing of the data in addition to running the models themselves. 
Included within the [workflow_examples](workflow_examples) folder of the repository are a number of example Snakemake workflow that show how to
run the entire process with a variety of models and end-goals. 

### To run the Snakemake workflow locally:

1. Install the dependencies in the `environment.yaml` file. With conda you can do this with `conda env create -f environment.yaml`
2. Activate your conda environment `source activate rdl_torch_tf`
3. Install the local `river-dl` package by `pip install path/to/river-dl/` (_optional_)
4. Edit the river-dl run configuration (including paths for I/O data) in the appropriate `config.yml`
from the [workflow_examples](workflow_examples) folder.
5. Run Snakemake with `snakemake --configfile config.yml -s Snakemake --cores <n>`

### To run the Snakemake Workflow on TallGrass
1. Request a GPU allocation and start an interactive shell

        salloc -N 1 -t 2:00:00 -p gpu -A <account> --gres=gpu:1 
        srun -A <account> --pty bash

2. Load the necessary cuda toolkit module and add paths to the cudnn drivers
        
        module load cuda11.3/toolkit/11.3.0 
        export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH
3. Follow steps 1-5 above as you would to run the workflow locally (note, you may need to change `tensorflow`
to `tensoflow-gpu` in the `environment.yml`). 

_After building your environment, you may want to make sure the recommended versions of PyTorch and CUDA were installed
according to the [PyTorch documentation](https://pytorch.org/). You can see the installed versions
by calling `conda list` within your activated environment._

### The data
The data used to run this model currently are specific to the Delaware River Basin but will soon be made more generic.

___

### Disclaimer
This software is in the public domain because it contains materials that originally came from the U.S. Geological Survey, an agency of the United States Department of Interior. For more information, see the official USGS copyright policy

Although this software program has been used by the U.S. Geological Survey (USGS), no warranty, expressed or implied, is made by the USGS or the U.S. Government as to the accuracy and functioning of the program and related program material nor shall the fact of distribution constitute any such warranty, and no responsibility is assumed by the USGS in connection therewith.

This software is provided “AS IS.”
