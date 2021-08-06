# Creating a consistent environment for for the River-dl repository.
We've set up a workflow to use Docker and Singularity images to standardize 
the environment across computing systems and users for river-dl.  The workflow uses a base docker image that
can be run locally on CPU or converted into a Singularity image on Tallgrass or other HPC system with access to 
GPU. For an introduction to Docker, check out the [Docker tutorial](https://docs.docker.com/get-started/). The following walks
through pulling the river-dl docker image locally and on Tallgrass as well as one example of how to run
the associated snakemake workflow.

### 1) Pull the Docker and run River-dl locally 
Pull the river-dl image from docker hub. Earlier versions can be pulled by
specifying the tag (i.e. `v1.1`) instead of using `latest` 

`docker pull simontopp/river-dl:latest`

Navigate to the river-dl repsository.  Once there, create a container from the image and open it
with the river-dl repository mounted as volume.

`docker run --entrypoint bash -it --mount "src=$(pwd),target=/river-dl,type=bind" -w '/river-dl' simontopp/river-dl`

Above, the `-it` flag creates an interactive container, `--mount` mounts the container to your current 
working directory (with read/write access), and `-w` sets the working directory within the docker container.

Once in the container run the snakemake file just as you would within terminal.

`snakemake --configfile config.yml --cores all`

If you get a cryptic `Killed` return, you may need to increase the memory usage allowed by docker. The
default on a mac is 2GB.  To change this, go to docker and select Settings > Resources and increase the allowed
memory.

Once your done, exit the container  by typing `exit`.

**After exiting, you can view the container ID by typing `docker container ls -a` and either remove it 
with `docker rm [container id]` or reactivate it with `docker start -i [container id]`. The above script recreates
the container each time, so if you don't remove it be aware you'll end up with multiple containers**

### 2) Convert the Docker image to a Singularity and run River-dl on Tallgrass.
_If you've already pulled the image to Tallgrass, skip to "Running River-dl in the singularity container"_

On Tallgrass, load singularity

`module load singularity`

Pull the river-dl image from dockerhub and convert it to a singularity image. Replace `latest` with desired tag.
The following pulls the image into the river-dl directory on Tallgrass and names the converted .sif image `riv-dl-sing.sif`.

`singularity pull ~/river-dl/riv-dl-sing.sif docker://simontopp/river-dl:latest`

_Running River-dl in the singularity container_

Allocate a GPU node and open a bash script within that node

`salloc -N 1 -n 1 -c 8 -p gpu -A [Account] -t 2:00:00 --gres=gpu:1`

`srun -A [Account] --pty bash`

Set up the environment and specify paths to necessary nvidia libraries

`module load singularity cuda10.0 cuda10.0/blas cuda10.1`

`export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_7.6.5/lib64:$LD_LIBRARY_PATH`

Move to the river-dl directory
`cd [path to river-dl]`

Start the singularity container and bind the necessary nvidia libraries and your river-dl directory.

`singularity shell --nv --bind /cm/local/apps,/cm/shared/apps,[path to river-dl]:/river-dl/ riv-dl-sing.sif`

Run the snakemake workflow

`snakemake --configfile config.yml --cores all`

### 3) Updating the docker image

If you need to update the container image for some reason (i.e. add an additional package), you'll need to re-build the image
locally on your lap-top or desktop using the Dockerfile in the base directory of river-dl.  To do this, open up the dockerfile and add the packages you need
to the list, then run: 

`docker build -t river-dl:v0.1 .`

Above, the `-t` flag is what allows you to name the container in the name:tag format.

Then push your updated image to docker hub (you'll need to make an account if you don't have one).

`docker push river-dl:v0.1`

At this point, follow this instruction in steps 1 and 2 to use your updated container.

### Planned additions:
Eventually this readme will include additional instructions for running interactive sessions through a jupyter notebook
and for submitting batch jobs within the river-dl Docker/Singularity container.