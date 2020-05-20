# Path Imputation Strategies for Signature Models
 
The goal of this project is to extend signature models to the real-world setting of classifying irregularly spaced and incompletely observed time series. 


## Environment
Please install the dependencies as indicated in the Pipfile
```> pipenv install --skip-lock```  
```> pipenv shell```

Note, that the newer GPytorch versions ( >1.0.0 ) tend to overwrite the torch version in pipenv. If this happens (e.g. with GPytorch 1.0.1), a working solution was to just ```>pip intall torch==1.2.0``` 
after the pipenv install

## Hypersearch Commands
### gpu scheduler
if you want to use a gpu scheduler, simply install this one via:
```> pip install simple_gpu_scheduler ```
### generate hypersearch commands:
To ensure that there is no later collision between processes trying to cache at the same time and place, first run the following bash script configured to preprocessing=true, 
it then genereates commands which will only handle the preprocessing and caching. 
```> ./scripts/generate_UEA_hypersearch_commands.sh ``` 
After preprocessing, deactivate it (setting preprocessing=false), and rerun the script.
This bash script generates multiple command files, one for GP-based methods and one for the other imputed methods (usually requiring less memory).
For instance, this one: ```command_UEA_imputed_hypersearches.txt```

### actually run the hypersearch via gpu scheduler on the first 3 devices of your server:
```> simple_gpu_scheduler --gpus 0,1,2 < command_UEA_imputed_hypersearches.txt ```


# Quick fitting, testing
## Train a end-to-end, posterior moments GP-imputed Signature Model, specifying signature depth (truncation level) to 3

```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```
```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=monte_carlo model.parameters.sig_depth=2```
```> python exp/train_model.py with model.GPGRUSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=2```

## Start a hyperparameter search: for the hypersearches, the models and datasets are defined (and extended with hyperparameter spaces) in /exp/hypersearch_configs.py

```>python exp/hyperparameter_search.py -F exp_runs/SignatureModel with GP_mom_SignatureModel Physionet2012 ```

### To check to current parameter configurations (handled via sacred), use print_config.
### For instance, to inspect all set parameters in one of the commands above, use:
```> python exp/train_model.py print_config with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```

