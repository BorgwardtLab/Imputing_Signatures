#!/bin/bash
device=cpu
gp_models=(GP_mc_SignatureModel GP_mom_SignatureModel GP_mc_GRUSignatureModel GP_mom_GRUSignatureModel GP_mom_GRUModel GP_mc_GRUModel) #GP_mom_DeepSignatureModel GP_mc_DeepSignatureModel)
output_pattern='experiments/hyperparameter_search/GP_based/{dataset}/{model}.json'
datasets=(PenDigits, .., ..)
#GP methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${gp_models[*]} \
  --name dataset --set ${datasets[*]} \
  --output-pattern ${output_pattern}
  --name dummy --set overrides.device=${device} \



imputed_models=(ImputedSignatureModel ImputedRNNSignatureModel ImputedRNNModel)
data_formats=(zero linear forwardfill causal indicator)
output_pattern='experiments/hyperparameter_search/{dataset}/{data_format}{model}.json'

#GP methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${imputed_models[*]} \
  --name dataset --set ${datasets[*]} \
  --name data_format \
  --set ${data_formats[*]} \
  --output-pattern ${output_pattern \
  --name dummy --set overrides.device=${device} \
  --name dummy --set overrides.model__parameters__output_device=${device} \


