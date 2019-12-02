#!/bin/sh

module load python/3.6.0
pipenv shell

index=93
thres=5 
#declare -a arr=("Sig_kNN" "Sig_LR" "Sig_LGBM" "DTW_kNN")
declare -a arr=("DTW_kNN")

for i in $(seq 0 $index); do  
   #for j in 5; do #{5,10,30,50} 
       for m in "${arr[@]}"; do 
            echo submitting job $i, $m
            bsub -n 4 -W 24:00 -R "rusage[mem=4048]" -e "clf_jobs/" -o "clf_jobs/" "python src/main.py --dataset $i --method $m --use_subsampling --thres $thres --interpol GP";  
        done;
    #done;
    for m in "${arr[@]}"; do 
        echo submitting job without subsampling for method $m
            bsub -n 4 -W 24:00 -R "rusage[mem=4048]" -e "clf_jobs/" -o "clf_jobs/" "python src/main.py --dataset $i --method $m";  
        done;
done

