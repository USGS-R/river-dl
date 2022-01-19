## Intro
The Snakefiles and associated config files in this directory are meant to provide examples of Snakemake workflows that use river-dl functionality. Snakemake is a very flexible and useful tool and these are just a few examples of how it can be used. For more information about Snakemake, please see the Snakemake docs which are very comprehensive and helpful. A link to the documents is below.

## Example workflows
### Snakefile_basic.smk, config_basic.yml
This is an example of a basic workflow with an LSTM and only one training phase.

### Snakefile_pretrain_LSTM.smk, config_pretrain_LSTM.yml
This is an example of a basic workflow with an LSTM with pretraining and finetuning.

### Snakefile_rgcn.smk, config_rgcn.yml
This is an example of a basic workflow with a RGCN and two training phases. It also uses early stopping.

### Snakefile_gw.smk, config_gw.yml
This is an example of a workflow that does some customization to include groundwater-specific considerations. It uses an RGCN and and two training phase (pretraining and finetuning). It also uses early stopping.

## Running a workflow
Assuming you have `snakemake` installed, you would run the "basic" workflow with 1 core with the following command: 

```
snakemake -s Snakefile_basic.smk --configfile config.yml -j1
```

## Links
Snakemake docs: [https://snakemake.readthedocs.io/en/stable/](https://snakemake.readthedocs.io/en/stable/)


