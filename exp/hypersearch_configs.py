def Physionet2012():
    overrides = {'dataset__name': 'Physionet2012'}

def add_datasets(experiment):
    experiment.named_config(Physionet2012)


def GP_mc_SignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 5),
        'model__parameters__extra_channels': ('Integer', 5, 15),
        'model__parameters__channel_groups': ('Integer', 1, 10)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mom_SignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 5),
        'model__parameters__extra_channels': ('Integer', 5, 15),
        'model__parameters__channel_groups': ('Integer', 1, 10)
    }
    overrides = {
        'model__name': 'GPSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mc_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 5),
        'model__parameters__extra_channels': ('Integer', 5, 15),
        'model__parameters__channel_groups': ('Integer', 1, 10),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPGRUSignatureModel',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_mom_GRUSignatureModel():
    train_module = 'train_model'
    hyperparameter_space = {   
        'model__parameters__sig_depth': ('Integer', 2, 3),
        'model__parameters__extra_channels': ('Integer', 5, 10),
        'model__parameters__channel_groups': ('Integer', 1, 4),
        'model__parameters__length': ('Integer', 3, 10),
        'model__parameters__rnn_channels': ('Categorical', [16,32,64,128]) 
    }
    overrides = {
        'model__name': 'GPGRUSignatureModel',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }


def GP_Sig():
    train_module = 'train_model'
    overrides = {
        'model__name': 'GP_Sig',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_Sig_Moments():
    train_module = 'train_model'
    overrides = {
        'model__name': 'GP_Sig',
        'model__parameters__sampling_type': 'moments',
        'model__parameters__n_mc_smps': 1,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }



def add_models(experiment):
    experiment.named_config(GP_Sig)
    experiment.named_config(GP_Sig_Moments)
    experiment.named_config(GP_mom_SignatureModel)
    experiment.named_config(GP_mc_SignatureModel)
    experiment.named_config(GP_mom_GRUSignatureModel)
    experiment.named_config(GP_mc_GRUSignatureModel)


