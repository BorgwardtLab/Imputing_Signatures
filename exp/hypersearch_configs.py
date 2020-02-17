def Physionet2012():
    overrides = {'dataset__name': 'Physionet2012'}

def PenDigits():
    overrides = {'dataset__name': 'PenDigits'}

def LSST():
    overrides = {'dataset__name': 'LSST'}

def CharacterTrajectories():
    overrides = {'dataset__name': 'CharacterTrajectories'}


def add_datasets(experiment):
    experiment.named_config(Physionet2012)
    experiment.named_config(PenDigits)
    experiment.named_config(LSST)
    experiment.named_config(CharacterTrajectories)

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

## Deep Signature Models
##TODO: overrides: use_constant_trick=False?
def GP_mc_DeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__hidden_channels1': ('Integer', 8, 32),
        'model__parameters__hidden_channels2': ('Integer', 4, 8),
        'model__parameters__kernel_size': ('Integer', 3, 6),
        'weight_decay': ('Real', 10**-4, 10**-3, 'uniform')
    }
    overrides = {
        'model__parameters__sig_depth': 2,
        'model__name': 'GPDeepSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mom_DeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__hidden_channels1': ('Integer', 8, 32),
        'model__parameters__hidden_channels2': ('Integer', 4, 8),
        'model__parameters__kernel_size': ('Integer', 3, 6),
    }
    overrides = {
        'model__parameters__sig_depth': 2,
        'model__name': 'GPDeepSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

## RNN Models
def GP_mom_GRUModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128,256,512]) 
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
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128,256,512]) 
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

def ImputedDeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__hidden_channels1': ('Integer', 8, 32),
        'model__parameters__hidden_channels2': ('Integer', 4, 8),
        'model__parameters__kernel_size': ('Integer', 3, 6),
        'batch_size': ('Categorical', [32,64,128,256])
    }
    overrides = {
        'model__name': 'ImputedDeepSignatureModel',
        'model__parameters__sig_depth': 2,
        'virtual_batch_size': None
    }

def ImputedRNNModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__hidden_size': ('Categorical', [16,32,64,128,256,512]),
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


####################
#Subsampling schemes
####################

def MissingAtRandomSubsampler():
    overrides = {
    'subsampler_name': 'MissingAtRandomSubsampler',
    'subsampler_parameters': { 'probability': 0.5  }
    }

def LabelBasedSubsampler():
    overrides = {
    'subsampler_name': 'LabelBasedSubsampler',
    'subsampler_parameters': { 'probability_ranges': [0.4, 0.6] }
    }


def add_models(experiment):
    experiment.named_config(GP_mom_SignatureModel)
    experiment.named_config(GP_mc_SignatureModel)
    experiment.named_config(GP_mom_GRUSignatureModel)
    experiment.named_config(GP_mc_GRUSignatureModel)
    experiment.named_config(GP_mom_GRUModel)
    experiment.named_config(GP_mc_GRUModel)
    experiment.named_config(GP_mc_DeepSignatureModel)
    experiment.named_config(GP_mom_DeepSignatureModel)

    experiment.named_config(ImputedSignatureModel)
    experiment.named_config(ImputedRNNSignatureModel)
    experiment.named_config(ImputedRNNModel)
    experiment.named_config(ImputedDeepSignatureModel)

    experiment.named_config(zero)
    experiment.named_config(linear)
    experiment.named_config(forwardfill)
    experiment.named_config(causal)
    experiment.named_config(indicator)

    experiment.named_config(MissingAtRandomSubsampler)
    experiment.named_config(LabelBasedSubsampler)


