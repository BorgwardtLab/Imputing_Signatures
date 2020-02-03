#!/bin/bash

#clf_models=(GP_mc_SignatureModel GP_mom_SignatureModel GP_mc_GRUSignatureModel GP_mom_GRUSignatureModel GP_mom_GRUModel GP_mc_GRUModel)
#
#output_pattern='experiments/hyperparameter_search/GP_based/{dataset}/{model}.json'
#
##GP methods:
#python scripts/configs_from_product.py exp.hyperparameter_search \
#  --name model \
#  --set ${clf_models[*]} \
#  --name dataset --set Physionet2012 \
#  --output-pattern ${output_pattern}
#


clf_models=(ImputedSignatureModel ImputedRNNSignatureModel ImputedRNNModel)
data_formats=(zero linear forwardfill causal indicator)
output_pattern='experiments/hyperparameter_search/{dataset}/{data_format}{model}.json'

#GP methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${clf_models[*]} \
  --name dataset --set Physionet2012 \
  --name data_format \
  --set ${data_formats[*]} \
  --output-pattern ${output_pattern}



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