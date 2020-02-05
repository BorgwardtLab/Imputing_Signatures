"""Module containing sacred functions for handling ML models."""
import inspect
from sacred import Ingredient

from src import models

ingredient = Ingredient('model')


@ingredient.config
def cfg():
    """Model configuration."""
    name = ''
    parameters = {
    }


@ingredient.named_config
def GPSignatureModel():
    """MGP Adapter with Signature Model (using Monte-carlo sampling)."""
    name = 'GPSignatureModel'
    parameters = {
        'sampling_type': 'monte_carlo' ,
        'n_mc_smps': 10,
        'n_devices': 1,
        'output_device': 'cuda'
    }

@ingredient.named_config
def ImputedSignatureModel():
    """Signature Model (requiring imputation!)."""
    name = 'ImputedSignatureModel'

@ingredient.named_config
def ImputedRNNSignatureModel():
    """RNNSignature Model (requiring imputation!)."""
    name = 'ImputedRNNSignatureModel'

@ingredient.named_config
def ImputedDeepSignatureModel():
    """DeepSignature Model (requiring imputation!)."""
    name = 'ImputedDeepSignatureModel'

@ingredient.named_config
def ImputedRNNModel():
    """RNN Model (requiring imputation!)."""
    name = 'ImputedRNNModel'
    parameters = {
        'rnn_type': 'gru' #lstm alternative
    }  

@ingredient.named_config
def GPRNNSignatureModel():
    """MGP Adapter with Signature Model (using Monte-carlo sampling)."""
    name = 'GPRNNSignatureModel'
    parameters = {
        'sampling_type': 'monte_carlo',
        'n_mc_smps': 10,
        'n_devices': 1,
        'output_device': 'cuda'
    }

@ingredient.named_config
def GPDeepSignatureModel():
    """MGP Adapter with Deep Signature Model (using Monte-carlo sampling)."""
    name = 'GPDeepSignatureModel'
    parameters = {
        'sampling_type': 'monte_carlo',
        'n_mc_smps': 10,
        'n_devices': 1,
        'output_device': 'cuda'
    }

@ingredient.named_config
def GPRNNModel():
    """MGP Adapter with RNN Model (using Monte-carlo sampling)."""
    name = 'GPRNNModel'
    parameters = {
        'sampling_type': 'monte_carlo',
        'n_mc_smps': 10,
        'n_devices': 1,
        'output_device': 'cuda'
    }

@ingredient.capture
def get_instance(n_input_dims, out_dimension, name, parameters, _log, _seed):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    # Get the mode class
    model_cls = getattr(models, name)

    # Inspect if the constructor specification fits with additional_parameters
    signature = inspect.signature(model_cls)
    available_parameters = signature.parameters
    for key in parameters.keys():
        if key not in available_parameters.keys():
            # If a parameter is defined which does not fit to the constructor
            # raise an error
            raise ValueError(
                f'{key} is not available in {name}\'s Constructor'
            )

    # Now check if optional parameters of the constructor are not defined
    optional_parameters = list(available_parameters.keys())[4:]
    for parameter_name in optional_parameters:
        # Copy list beforehand, so we can manipulate the parameter dict in the
        # loop
        parameter_keys = list(parameters.keys())
        if parameter_name not in parameter_keys:
            if parameter_name != 'random_state':
                # If an optional parameter is not defined warn and run with
                # default
                default = available_parameters[parameter_name].default
                _log.warning(
                    f'Optional parameter {parameter_name} not explicitly '
                    f'defined, will run with {parameter_name}={default}'
                )
            else:
                _log.info(
                    f'Passing seed of experiment to model parameter '
                    '`random_state`.'
                )
                parameters['random_state'] = _seed

    return model_cls(n_input_dims, out_dimension, **parameters)
