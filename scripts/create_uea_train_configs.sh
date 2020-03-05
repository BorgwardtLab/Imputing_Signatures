#!/bin/bash

#FOR LEOMED this bit must run on GPU
device=cuda
gp_models=(model.GP_mc_SignatureModel)
output_pattern='experiments/train_uea_model/GP_based/{dataset}/{subsampler_name}/{subsampler_thresholds}/{imputation_scheme}/{model}.json'
thresholds=(subsampler_parameters.probability=0.5 subsampler_parameters.probability=0.75)
imputation_scheme=(imputation_scheme=zero imputation_scheme=forward_fill)
#GP methods:
python scripts/configs_from_product.py exp.train_uea_model \
  --separator '=' \
  --name model \
  --set ${gp_models[*]} \
  --name dataset --set dataset.PenDigits \
  --name subsampler_name \
  --set subsampler_name=MissingAtRandomSubsampler \
  --name subsampler_thresholds \
  --set ${thresholds[*]} \
  --name imputation_scheme \
  --set ${imputation_scheme[*]} \
  --output-pattern ${output_pattern}
  #--name dummy --set overrides.device=${device} \

exit

### CPU/or GPU JOBS:
#FOR EULER!
device=cpu
imputed_models=(ImputedSignatureModel ImputedRNNSignatureModel ImputedRNNModel ImputedDeepSignatureModel)
data_formats=(zero linear forwardfill causal indicator)
output_pattern='experiments/train_uea_model/{dataset}/{data_format}{model}.json'

#GP methods:
python scripts/configs_from_product.py exp.train_uea_model \
  --name model \
  --set ${imputed_models[*]} \
  --name dataset --set ${datasets[*]} \
  --name data_format \
  --set ${data_formats[*]} \
  --output-pattern ${output_pattern \
  --name dummy --set overrides.device=${device} \


