def Physionet2012():
    overrides = {'dataset__name': 'Physionet2012'}

def PenDigits():
    overrides = {
            'dataset__name': 'UEADataset',
            'dataset__parameters__dataset_name': 'PenDigits',
            'dataset__parameters__use_disk_cache': True }

def LSST():
    overrides = {
            'dataset__name': 'UEADataset',
            'dataset__parameters__dataset_name': 'LSST',
            'dataset__parameters__use_disk_cache': True }

def CharacterTrajectories():
    overrides = {
            'dataset__name': 'UEADataset',
            'dataset__parameters__dataset_name': 'CharacterTrajectories',
            'dataset__parameters__use_disk_cache': True }

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
        'model__parameters__channel_groups': ('Integer', 1, 5)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__final_network': [30,30],
        'model__parameters__include_original': False
    }

def GP_mom_SignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 5)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__final_network': [30,30],
        'model__parameters__include_original': False
    }

## RNN Signature Models

def GP_mc_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 5),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]),
    }
    overrides = {
        'model__name': 'GPRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__include_original': False
    }

def GP_mom_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 5),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__include_original': False
    }

## Deep Signature Models
##TODO: overrides: use_constant_trick=False?
def GP_mc_DeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__hidden_channels1': ('Integer', 4, 12),
        'model__parameters__hidden_channels2': ('Integer', 4, 12),
        'model__parameters__kernel_size': ('Integer', 3, 6),
    }
    overrides = {
        'model__name': 'GPDeepSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__include_original': False,
        'model__parameters__batch_norm': True 
    }

def GP_mom_DeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__hidden_channels1': ('Integer', 4, 12),
        'model__parameters__hidden_channels2': ('Integer', 4, 12,
        'model__parameters__kernel_size': ('Integer', 3, 6),
    }
    overrides = {
        'model__name': 'GPDeepSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda:0',
        'model__parameters__include_original': False,
        'model__parameters__batch_norm': True 
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
        'model__parameters__output_device': 'cuda:0',
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
        'model__parameters__output_device': 'cuda:0'
    }

###################################################
# Imputation-based Models (preprocessed imputation)
###################################################

def ImputedSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 5),
        'batch_size': ('Categorical', [32,64,128,256]) 
    }
    overrides = {
        'model__name': 'ImputedSignatureModel',
        'model__parameters__final_network': [30,30],
        'virtual_batch_size': None,
        'model__parameters__include_original': False
    }

def ImputedRNNSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 5),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]),
        'batch_size': ('Categorical', [32,64,128,256]) 
    }
    overrides = {
        'model__name': 'ImputedRNNSignatureModel',
        'model__parameters__rnn_type': 'gru',
        'virtual_batch_size': None,
        'model__parameters__include_original': False
    }

def ImputedDeepSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {
        'model__parameters__sig_depth': ('Integer', 2, 4),
        'model__parameters__hidden_channels1': ('Integer', 4, 12),
        'model__parameters__hidden_channels2': ('Integer', 4, 12),
        'model__parameters__kernel_size': ('Integer', 3, 6),
        'batch_size': ('Categorical', [32,64,128,256])
    }
    overrides = {
        'model__name': 'ImputedDeepSignatureModel',
        'virtual_batch_size': None,
        'model__parameters__include_original': False,
        'model__parameters__batch_norm': True 
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

################################################################
# Due to very short series, use custom subsampling for PenDigits:
################################################################
def MissingAtRandomSubsamplerPenDigits():
    overrides = {
    'subsampler_name': 'MissingAtRandomSubsampler',
    'subsampler_parameters': { 'probability': 0.3  }
    }

def LabelBasedSubsamplerPenDigits():
    overrides = {
    'subsampler_name': 'LabelBasedSubsampler',
    'subsampler_parameters': { 'probability_ranges': [0.2, 0.4] }
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
    experiment.named_config(MissingAtRandomSubsamplerPenDigits)
    experiment.named_config(LabelBasedSubsamplerPenDigits)


