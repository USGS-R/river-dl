# syntax=docker/dockerfile:1
FROM python:3.6

##RUN apt-get update && apt-get install -y python3.6 python3-pip
RUN python3.6 -m pip install --upgrade pip

#add additional Python packages to install here:
RUN pip3 install dask \
        jupyterlab \
        matplotlib \
        pandas \
        pyarrow \
        scikit-learn \
        snakemake \
        xarray==0.16.2 \
        zarr \
        statsmodels \
        seaborn \
        tensorflow==2.1.0 \
        tensorflow-gpu==2.1.0 \
        tensorflow-estimator==2.1.0
