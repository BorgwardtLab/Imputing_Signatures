def Physionet2012():
    overrides = {'dataset__name': 'Physionet2012'}

def add_datasets(experiment):
    experiment.named_config(Physionet2012)

def GP_Sig():
    train_module = 'train_model'
    #hyperparameter_space = {
    #    'batch_size': ('Integer', 16, 128)
    #}
    overrides = {
        'model__name': 'GP_Sig',
        'model__parameters__sampling_type': 'monte_carlo',
        'model__parameters__n_mc_smps': 10,
        'model__parameters__n_devices': 1,
        'model__parameters__output_device': 'cuda'
    }

def GP_Sig_Moments():
    train_module = 'train_model'
    #hyperparameter_space = {
    #    'batch_size': ('Integer', 16, 128)
    #}
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


