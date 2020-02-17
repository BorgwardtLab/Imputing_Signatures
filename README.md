# GP Signatures

The goal of this project is to extend signature models to the real-world setting of classifying irregularly spaced and incompletely observed time series. 
Specifically, we propose a GP-Sig, an end-to-end MGP adapter employing a deep signature model.

## Environment
Please install the dependencies as indicated in the Pipfile
```> pipenv install --skip-lock```  
```> pipenv shell```

Note, that the newer GPytorch versions ( >1.0.0 ) tend to overwrite the torch version in pipenv. If this happens (e.g. with GPytorch 1.0.1), a working solution was to just >pip intall torch==1.2.0 
after the pipenv install

## Train a end-to-end, posterior moments GP-imputed Signature Model, specifying signature depth (truncation level) to 3

```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```
```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=monte_carlo model.parameters.sig_depth=2```
```> python exp/train_model.py with model.GPGRUSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=2```

## Start a hyperparameter search: for the hypersearches, the models and datasets are defined (and extended with hyperparameter spaces) in /exp/hypersearch_configs.py

```>python exp/hyperparameter_search.py -F exp_runs/SignatureModel with GP_mom_SignatureModel Physionet2012 ```

### To check to current parameter configurations (handled via sacred), use print_config.
### For instance, to inspect all set parameters in one of the commands above, use:
```> python exp/train_model.py print_config with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```

