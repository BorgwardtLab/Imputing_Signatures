# GP Signatures

The goal of this project is to extend signature models to the real-world setting of classifying irregularly spaced and incompletely observed time series. 
Specifically, we propose a GP-Sig, an end-to-end MGP adapter employing a deep signature model.

## Environment
Please install the dependencies as indicated in the Pipfile
```> pipenv install --skip-lock```  
```> pipenv shell```


## Train a end-to-end, posterior moments GP-imputed Signature Model, specifying signature depth (truncation level) to 3

```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```
```> python exp/train_model.py with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=monte_carlo model.parameters.sig_depth=2```
```> python exp/train_model.py with model.GPGRUSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=2```

## Start a hyperparameter search: for the hypersearches, the models and datasets are defined (and extended with hyperparameter spaces) in /exp/hypersearch_configs.py

```>python exp/hyperparameter_search.py -F exp_runs/SignatureModel with GP_mom_SignatureModel Physionet2012 ```

### To check to current parameter configurations (handled via sacred), use print_config.
### For instance, to inspect all set parameters in one of the commands above, use:
```> python exp/train_model.py print_config with model.GPSignatureModel dataset.Physionet2012 model.parameters.sampling_type=moments model.parameters.sig_depth=3```




## Below instructions are from an early phase of the projects and not guaranteed to still work:
### Hadamard MGP Adapter with off-the-shelf Deep Signature Model on Synthetic Data:
```> python scripts/test_mgp_adapter.py```

For quickly testing this on cpu (instead of gpu) use:
```> python scripts/test_mgp_adapter.py --device cpu```

### First script running on Physionet2012:
```> python scripts/mgp_adapter_physionet2012.py```


