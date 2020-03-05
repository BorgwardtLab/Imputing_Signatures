#!/bin/bash

exp_dir=exp
fit_module=hyperparameter_search
fit_module_path=${exp_dir}/${fit_module}.py

datasets=(Physionet2012)
gp_models=(GP_mc_SignatureModel GP_mom_SignatureModel GP_mc_GRUSignatureModel GP_mom_GRUSignatureModel GP_mom_GRUModel GP_mc_GRUModel) #GP_mom_DeepSignatureModel GP_mc_DeepSignatureModel
imputed_models=(ImputedSignatureModel ImputedRNNSignatureModel ImputedRNNModel) #ImputedDeepSignatureModel
data_formats=(zero linear forwardfill causal indicator) #only for imputed models
model_types=(GP imputed) #we distinguish between those two types of models
preprocessing=false

# Create command files for both model types seperately, GP and imputed models
for model_type in ${model_types[*]}; do
    for dataset in ${datasets[*]}; do
        if [ "$preprocessing" == "true" ]; then
            outfile=scripts/command_${dataset}_${model_type}_preprocessing.txt
        else
            outfile=scripts/command_${dataset}_${model_type}_hypersearches.txt 
        fi  
        
        #Remove existing outfile to not append commands there..
        if [ -f "$outfile" ]; then
            echo 'Out file already exists, removing it to create a new one..'
            rm $outfile
        fi

        #########################################################
        #First: Loop over datasets, models, and subsampling type:
        #########################################################

        # to this end, determine which models to use:
        if [ "$model_type" == "GP" ]; then 
            models=${gp_models[*]}
        else
            models=${imputed_models[*]}
            
        fi

        echo 'Looping over' ${model_type} 'models:'
        echo $models
 
        for model in ${models[*]}; do
            if [ "$preprocessing" == "true" ]; then
                #only start all imputation schemes as preprocessing (for one model)
                if [ "$model" == "ImputedSignatureModel" ]; then
                    #we need additional loop over imputation strategies
                    for data_format in ${data_formats[*]}; do
                        #write command to outfile, this one is just to run the preprocessing! 
                        echo python $fit_module_path with $model $dataset $data_format n_calls=1 n_random_starts=1 overrides.n_epochs=1 >> $outfile
                    done
                fi
            else
                if [ "$model_type" == "imputed" ]; then
                    #for imputed models, we need additional loop over imputation strategies
                    for data_format in ${data_formats[*]}; do
                        #define output directory of current hypersearch experiment
                        outdir=experiments/${fit_module}/${dataset}/${data_format}${model}
                        #write python command to outfile 
                        echo python $fit_module_path -F $outdir with $model $dataset $data_format >> $outfile
                    done
                else
                    #define output directory of current hypersearch experiment
                    outdir=experiments/${fit_module}/${dataset}/${model}
                    #write python command to outfile
                    echo python $fit_module_path -F $outdir with $model $dataset >> $outfile
                fi
            fi
        done
    done
done
    
