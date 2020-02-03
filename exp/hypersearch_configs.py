def Physionet2012():
    overrides = {'dataset__name': 'Physionet2012'}

def add_datasets(experiment):
    experiment.named_config(Physionet2012)

#######################################
# GP-Based Models
#######################################

## Simple Signature Models:

def GP_mc_SignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda',
        'model__parameters__final_network': [30,30]
    }

def GP_mom_SignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda',
        'model__parameters__final_network': [30,30]
    }

## RNN Signature Models

def GP_mc_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]),
        'weight_decay': ('Real', 10**-4, 10**-3, 'uniform')
    }
    overrides = {
        'model__name': 'GPRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mom_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

## RNN Models
def GP_mom_GRUModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPRNNModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mc_GRUModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPRNNModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

###################################################
# Imputation-based Models (preprocessed imputation)
###################################################

def ImputedSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10),
        'batch_size': ('Categorical', [32,64,128,256]) 
    }
    overrides = {
        'model__name': 'ImputedSignatureModel',
        'model__parameters__final_network': [30,30],
        'virtual_batch_size': None
    }

def ImputedRNNSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 10),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]),
        'batch_size': ('Categorical', [32,64,128,256]) 
    }
    overrides = {
        'model__name': 'ImputedRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'virtual_batch_size': None
    }

def ImputedRNNModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128]),
        'batch_size': ('Categorical', [32,64,128,256]) 
    }
    overrides = {
        'model__name': 'ImputedRNNModel',
        'model__parameters__rnn_type': 'gru',
        'virtual_batch_size': None
    }

###################
#imputation schemes
###################
def zero():
    overrides = {'data_format': 'zero'}
def linear():
    overrides = {'data_format': 'linear'}
def forwardfill():
    overrides = {'data_format': 'forwardfill'}
def causal():
    overrides = {'data_format': 'causal'}
def indicator():
    overrides = {'data_format': 'indicator'}


def add_models(experiment):
    experiment.named_config(GP_mom_SignatureModel)
    experiment.named_config(GP_mc_SignatureModel)
    experiment.named_config(GP_mom_GRUSignatureModel)
    experiment.named_config(GP_mc_GRUSignatureModel)
    experiment.named_config(GP_mom_GRUModel)
    experiment.named_config(GP_mc_GRUModel)
    experiment.named_config(ImputedSignatureModel)
    experiment.named_config(ImputedRNNSignatureModel)
    experiment.named_config(ImputedRNNModel)
    experiment.named_config(zero)
    experiment.named_config(linear)
    experiment.named_config(forwardfill)
    experiment.named_config(causal)
    experiment.named_config(indicator)

