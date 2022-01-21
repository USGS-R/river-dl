## Intro
The Snakefiles and associated config files in this directory are meant to provide examples of Snakemake workflows that use river-dl functionality. Snakemake is a very flexible and useful tool and these are just a few examples of how it can be used. For more information about Snakemake, please see the Snakemake docs which are very comprehensive and helpful. A link to the documents is below.

Hopefully these can serve as examples, but also hopefully you are thinking about what works best for _your_ follow or depart from these examples accordingly:
- You will need to modify some things if you want to use these workflows for your own purposes. See "What might change when you make your own workflow" section below for more details on what and where you might modify.
- There are _many_ ways that snakemake could be used with river-dl to execute these workflows. If you want to learn more about the basics of snakemake, we recommend checking out their [tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html) and their [comprehensive docs](https://snakemake.readthedocs.io/en/stable/).
- There are also many ways that river-dl can be used _without_ snakemake. For example, you could simply use a Python file to call different river-dl functions. This option generally trades robustness and scalability for expediency.

## Example Snakemake workflows
### Snakefile_basic.smk, config_basic.yml
This is an example of a basic workflow with an LSTM and only one training phase.

### Snakefile_pretrain_LSTM.smk, config_pretrain_LSTM.yml
This is an example of a basic workflow with an LSTM with pretraining and finetuning.

### Snakefile_rgcn.smk, config_rgcn.yml
This is an example of a basic workflow with a RGCN and two training phases. It also uses early stopping.

### Snakefile_gw.smk, config_gw.yml
This is an example of a workflow that does some customization to include groundwater-specific considerations. It uses an RGCN and and two training phase (pretraining and finetuning). It also uses early stopping.

## Running an example workflow
Assuming you have `snakemake` installed, you would run the "basic" workflow with 1 core with the following command: 

```
snakemake -s Snakefile_basic.smk --configfile config.yml -j1
```

## Train a model on a GPU
To train a model using GPU, this line can be added to the `run` part of your training rule before the actual code that trains the model:
```python
os.system("module load analytics cuda10.1/toolkit/10.1.105"
```

## What might change when you make your own workflow
If you want to build from one of the example workflows, you will _need_ to change some things and will likely _want_ to change others. Most of the things you will need or want to change will be changed in the `config.yml` file.

### Changes that are made in `config.yml` 

**Things you will most likely _need_ to change:**
- the input datasets
    - the data used in the example workflows is a very small subset of meteorological drivers and water temperature and streamflow observations. These are defined in the `sntemp_file` and `obs_file`values in the `"*_config.yml"` files, respectively.
    - you will need to replace those values in the config file with the filepaths to the data files that contain the sites and times that you are interested in modeling
- the train/validation/test dates
    - you will likely need to change these in the config.yml file as well to match the time periods you are wanting to use. Note - the river-dl preprocessing function `prep_all_data` can take multple, discontinuous time periods for any of the train, validation, and test partitions
- The number of epochs
	- the example workflows are only for 5 epochs. We'll assume you'll want to train for more than that :)


**Things you may _want_ to change:**
- `y_vars` - if you're output variables are different
- `out_dir` - so you're outputs are going where you want them to
- other hyperparameters - such as (`dropout`, `hidden_size`, and learning rates)


### Changes that are made `Snakefile` 
**Things you will most likely _need_ to change (or at least make sure it's what you want): **
- the deep learning `model` you are wanting to train/evaluate. This is defined and compiled (with the optimizer and loss function) in the Snakefile. For example, this happens on line 65 of `Snakefile_basic.smk`:
```python
model = LSTMModel(
    config['hidden_size'],
    recurrent_dropout=config['recurrent_dropout'],
    dropout=config['dropout'],
    num_tasks=len(config['y_vars'])
)
```
