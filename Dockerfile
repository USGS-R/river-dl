# syntax=docker/dockerfile:1
## This base image will read the GPU drivers when the container is being built
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

RUN apt-get update && apt-get install -y python3.6 python3-pip
RUN python3.6 -m pip install --upgrade pip

#add additional Python packages to install here:
RUN pip3 install dask \
        jupyterlab \
        matplotlib \
        pandas \
        pyarrow \
        scikit-learn \
        snakemake \
        xarray \
        zarr \
        tensorflow==2.1.0 \
        tensorflow-gpu==2.1.0 \
        tensorflow-estimator==2.1.0
