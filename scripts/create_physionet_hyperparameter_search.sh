#!/bin/bash

n_runs=5
#gp_models=(GP_mc_SignatureModel GP_mom_SignatureModel GP_mc_GRUSignatureModel GP_mom_GRUSignatureModel GP_mom_GRUModel GP_mc_GRUModel)
#output_pattern='experiments/hyperparameter_search/{dataset}/{model}.json'
#
##GP methods:
#python scripts/configs_from_product.py exp.hyperparameter_search \
#  --name model \
#  --set ${gp_models[*]} \
#  --name dataset --set Physionet2012 \
#  --output-pattern ${output_pattern}
#
#gp_models=(GP_mom_DeepSignatureModel GP_mc_DeepSignatureModel)
#output_pattern='experiments/hyperparameter_search/{dataset}/{model}.json'
#
#
##GP DeepSig methods:
#python scripts/configs_from_product.py exp.hyperparameter_search \
#  --name model \
#  --set ${gp_models[*]} \
#  --name dataset --set Physionet2012 \
#  --output-pattern ${output_pattern} \
#  --name dummy --set overrides.model__parameters__kernel_size=1
## USE THIS LAST PART ONLY FOR GP DEEPSIG!
#
#

imputed_models=(ImputedSignatureModel ImputedRNNSignatureModel ImputedRNNModel)
data_formats=(zero linear forwardfill causal indicator)
output_pattern='experiments/cleaned/hyperparameter_search/{dataset}/{data_format}{model}.json'

#GP methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${imputed_models[*]} \
  --name dataset --set Physionet2012 \
  --name data_format \
  --set ${data_formats[*]} \
  --output-pattern ${output_pattern} \
  --name dummy --set n_calls=$n_runs \
  --name dummy --set n_random_calls=$n_runs 
  #--name n_calls --set $n_runs \
  #--name n_random_starts --set $n_runs

imputed_models=(ImputedDeepSignatureModel)
data_formats=(zero linear forwardfill causal indicator)
output_pattern='experiments/cleaned/hyperparameter_search/{dataset}/{data_format}{model}.json'

#GP DeepSig methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${imputed_models[*]} \
  --name dataset --set Physionet2012 \
  --name data_format \
  --set ${data_formats[*]} \
  --output-pattern ${output_pattern} \
  --name dummy --set overrides.model__parameters__kernel_size=1 \
  --name dummy --set n_calls=$n_runs \
  --name dummy --set n_random_calls=$n_runs 

# USE THIS LAST PART ONLY FOR GP DEEPSIG!


#python scripts/configs_from_product.py exp.hyperparameter_search \
#    --name model \
#    --set ${ae_models[*]} \
#    --name dataset --set CIFAR \
#    --name dummy --set overrides.model__parameters__autoencoder_model=DeepAE \
#    --name dummy --set overrides.model__parameters__ae_kwargs__input_dims=${input_dims} \
#    --output-pattern ${output_pattern}
#
##VAE method:
#python scripts/configs_from_product.py exp.hyperparameter_search \
#  --name model \
#  --set Vanilla \
#  --name dataset --set MNIST FashionMNIST \
#  --name dummy --set overrides.model__parameters__autoencoder_model=DeepVAE \
#  --output-pattern ${output_pattern_vae}
#
#python scripts/configs_from_product.py exp.hyperparameter_search \
#    --name model \
#    --set Vanilla \
#    --name dataset --set CIFAR \
#    --name dummy --set overrides.model__parameters__autoencoder_model=DeepVAE \
#    --name dummy --set overrides.model__parameters__ae_kwargs__input_dims=${input_dims} \
#    --output-pattern ${output_pattern_vae}
#
##Classic, non-deep Baselines: 
#python scripts/configs_from_product.py exp.hyperparameter_search \
#  --name model \
#  --set ${competitor_methods[*]} \
#  --name dataset --set MNIST FashionMNIST CIFAR \
#  --output-pattern ${output_pattern}
