# GP Signatures

The goal of this project is to extend signature models to the real-world setting of classifying irregularly spaced and incompletely observed time series. 
Specifically, we propose a GP-Sig, an end-to-end MGP adapter employing a deep signature model.

## Environment
Please install the dependencies as indicated in the Pipfile
```> pipenv install --skip-lock```  
```> pipenv shell```

## Hadamard MGP Adapter with off-the-shelf Deep Signature Model on Synthetic Data:
```> python scripts/test_mgp_adapter.py```

For quickly testing this on cpu (instead of gpu) use:
```> python scripts/test_mgp_adapter.py --device cpu```

## First script running on Physionet2012:
```> python scripts/mgp_adapter_physionet2012.py```


