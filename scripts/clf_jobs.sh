#!/bin/sh

module load python/3.6.0
pipenv shell

index=93

for i in $(seq 0 $index); do  
   for j in {5}; do #{5,10,30,50} 
       for m in {Sig_kNN, Sig_LR, Sig_LGBM, DTW_kNN}; do 
            echo submitting job $i, $j, $m
            bsub -n 2 -W 2:00 -R "rusage[mem=4048]" -e "clf_jobs/" -o "clf_jobs/" "python src/main.py --dataset $i --method $m --use_subsampling --thres $j --interpol GP";  
        done;
    done;
    for m in {Sig_kNN, Sig_LR, Sig_LGBM, DTW_kNN}; do 
        echo submitting job without subsampling for method $m
            bsub -n 2 -W 2:00 -R "rusage[mem=4048]" -e "clf_jobs/" -o "clf_jobs/" "python src/main.py --dataset $i --method $m --interpol GP";  
        done;
done

