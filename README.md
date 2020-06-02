# Path Imputation Strategies for Signature Models of Irregularly Time Series
 
Anonymous code submission. Below commands assume that a GPU is available.

## Environment
Please install the dependencies as indicated in the Pipfile 
```> pipenv install --skip-lock```  
```> pipenv shell```

Note, that the newer GPytorch versions ( >1.0.0 ) tend to overwrite the torch version in pipenv. If this happens (e.g. with GPytorch 1.0.1), a working solution was to just ```>pip intall torch==1.2.0``` 
after the pipenv install

## Setting up data:
The physionet 2012 dataset has to be downloaded with the following shell script:  
```>source data/physionet_2012/download.sh```   
The other datasets are already in data/Multivariate_ts

## Hypersearch Commands
### gpu scheduler
if you want to use a gpu scheduler, simply install this one via:
```> pip install simple_gpu_scheduler ```

### generate hypersearch commands:
```> python scripts/generate_hypersearch_commands.py```

This script generates multiple command files, one for GP-based methods and one for the other imputed methods (usually requiring less memory).
For instance, this one: ```scripts/commands/command_LSST_imputed_hypersearches.csv```

### actually run the hypersearch via gpu scheduler on the first 3 devices of your server:
```> simple_gpu_scheduler --gpus 0,1,2 < command_LSST_imputed_hypersearches.csv ```

If there is a configuration problem with the virtual environment and the gpu scheduler, alternatively those python commands could be started manually
or sequentially via  
```> source scripts/commands/command_LSST_imputed_hypersearches.csv```  

### After having run a hyperparameter search, create repetitions:
```> python scripts/generate_repetitions.py``` 
This script assumes that the results of the hyperparameter search are stored in experiments/hyperparameter_search 
Again, as with the hyperparameter search:
```> simple_gpu_scheduler --gpus 0,1,2 < command_LSST_imputed_repetitions.csv ```

## Quick fitting, testing
### Train a end-to-end, posterior moments GP-imputed Signature Model, specifying signature depth (truncation level) to 3

```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```  
```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=monte_carlo model.parameters.sig_depth=2```  
```> python exp/train_model.py with model.GPGRUSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=2```  

### Manually start one hyperparameter search: for the hypersearches, the models and datasets are defined (and extended with hyperparameter spaces) in /exp/hypersearch_configs.py

```>python exp/hyperparameter_search.py -F exp_runs/SignatureModel with GP_mom_SignatureModel Physionet2012 ```

### To check to current parameter configurations (handled via sacred), use print_config.
### For instance, to inspect all set parameters in one of the commands above, use:
```> python exp/train_model.py print_config with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```

# Paper configurations  
The configurations used in the paper (repetition configs as determined by hyperparameter search), are accessible in the path `experiments/train_model`  
Train a model with a stored ```config.json```:  
```> python exp/train_model.py with path/to/config.json```

