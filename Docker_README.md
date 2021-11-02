# Creating a consistent environment for for the River-dl repository.
We've set up a workflow to use Docker and Singularity images to standardize 
the environment across computing systems and users for river-dl.  The workflow uses a base docker image that
can be run locally on CPU or converted into a Singularity image on Tallgrass or other HPC system with access to 
GPU. For an introduction to Docker, check out the [Docker tutorial](https://docs.docker.com/get-started/). The following walks
through pulling the river-dl docker image locally and on Tallgrass as well as two examples of how to run
the associated snakemake workflow.

### 1) Pull the Docker image and run River-dl locally 
Pull the river-dl image from Gitlab. Versions can be specified by
adding the version tag (i.e. `v1.1`) after river-dl in the format `image:tag`.
By default, it will pull `river-dl:latest`.

`docker pull code.chs.usgs.gov:5001/wma/wp/river-dl`

Navigate to the river-dl repsository.  Once there, create a container from the image and open it
with the river-dl repository mounted as volume.

`docker run --entrypoint bash -it --mount "src=$(pwd),target=/river-dl,type=bind" -w '/river-dl' river-dl`

Above, the `-it` flag creates an interactive container, `--mount` mounts the container to your current 
working directory (with read/write access), and `-w` sets the working directory within the docker container.

Once in the container, run the snakemake file just as you would within terminal.

`snakemake --configfile config.yml --cores all`

If you get a cryptic `Killed` return, you may need to increase the memory usage allowed by docker. The
default on a mac is 2GB.  To change this, go to docker and select Settings > Resources and increase the allowed
memory.

Once your done, exit the container  by typing `exit`.

**After exiting, you can view the container ID by typing `docker container ls -a` and either remove it 
with `docker rm [container id]` or reactivate it with `docker start -i [container id]`. The above script recreates
the container each time, so if you don't remove it be aware you'll end up with multiple containers**

### 2) Convert the Docker image to a Singularity image and run River-dl on Tallgrass.

_If you've already pulled the image to Tallgrass, skip to "2.1) Running River-dl in the singularity container"_

On Tallgrass, load singularity

`module load singularity`

Navigate to your river-dl directory

`cd river-dl/`

Pull the river-dl container image from GitLab and convert it to a singularity image. Versions can be specified by
adding the version tag (i.e. `v1.1`) after river-dl in the format `image:tag`.

`singularity pull --docker-login docker://code.chs.usgs.gov:5001/wma/wp/river-dl`

You should now have a `.sif` file in the river-dl directory.  This is the singularity image.

#### 2.1) _Running River-dl in terminal within the singularity container_

Allocate a GPU node and open a bash script within that node

`salloc -N 1 -n 1 -c 8 -p gpu -A <Account> -t 2:00:00 --gres=gpu:1`

`srun -A <Account> --pty bash`

Set up the environment and specify paths to necessary nvidia libraries

`module load singularity cuda10.0 cuda10.0/blas cuda10.1`

`export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_7.6.5/lib64:$LD_LIBRARY_PATH`

Move to the river-dl directory
`cd <path to river-dl>`

Start the singularity container and bind the necessary nvidia libraries and your river-dl directory.

`singularity shell --nv --bind /cm/local/apps,/cm/shared/apps,[path to river-dl]:/river-dl/ river-dl_latest.sif`

Run the snakemake workflow

`snakemake --configfile config.yml --cores 1`

_Currently, running the snakemake with more than 1 core causes tensorflow to overload the GPU memory.  We think this is because
snakemake is trying to allocate the entire memory of the GPU to each task/cpu. We're working on specifying computing 
resources in the snakefile to avoid this, but it will likely require larger modifications to the pipeline._

#### 2.2) _Submitting a batch script to run River-dl within the singularity container_

You can also submit a batch script that will run the snakemake pipeline within your river-dl container.  An example
slurm file is below:


    #!/bin/bash
    #SBATCH -J train_riv_dl_sing
    #SBATCH -t 4:00:00   # time
    #SBATCH -o tmp/snakemake-%j.out
    #SBATCH -p gpu
    #SBATCH -A watertemp          # your account code
    #SBATCH -n 1
    #SBATCH -c 16
    #SBATCH -N 1
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32GB
        
     
    module load singularity cuda10.0 cuda10.0/blas cuda10.1
    export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_7.6.5/lib64:$LD_LIBRARY_PATH
    srun singularity exec --nv --bind /cm/local/apps,/cm/shared/apps,/home/stopp/river-dl:/river-dl river-dl_latest.sif snakemake --configfile config.yml --cores all --rerun-incomplete'

### 3) Updating the docker image
First, you'll need to create an access token to push the updated container to the Gitlab registry.
  * Create your access token on [Gitlab](https://code.chs.usgs.gov/-/profile/personal_access_tokens)
  * [Authenticate gitlab](https://docs.gitlab.com/ee/user/packages/container_registry/#authenticate-with-the-container-registry)
    using your access token by running the following command and entering your generated token when
    prompted for your password.
    
    `docker login code.chs.usgs.gov:5001 -u <username>`

To update the container image (i.e. add an additional package), you'll need to re-build the image
locally on your laptop or desktop using the Dockerfile in the base directory of river-dl. Then you'll need to push the new
image to the container repository on Gitlab. To do this, open up the dockerfile and add the packages you need
to the list, then build the container by running the following and replacing the tag with your version
tag: 

`docker build -t code.chs.usgs.gov:5001/wma/wp/river-dl:<tag> .`

Push the rebuilt container to Gitlab

`docker push code.chs.usgs.gov:5001/wma/wp/river-dl:<tag>`

At this point, repull the new container/tag and follow instructions in steps 1 and 2 to run the container.

