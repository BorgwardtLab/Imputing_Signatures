#!/bin/sh

module load python/3.6.0
pipenv shell

index=93

#python src/preprocessing.py --dataset $index

for i in $(seq 0 $index); do  
   for j in {5,10,30,50}; do #$(seq 0 20 80) 
        #for k in {GP,linear}; do
            echo submitting job $i, $j
            bsub -n 2 -W 1:00 -R "rusage[mem=1048]" -e "jobs/" -o "jobs/" "python src/preprocessing.py --dataset $i --thres $j --interpol linear";  
            bsub -n 2 -W 8:00 -R "rusage[mem=4048]" -e "jobs/" -o "jobs/" "python src/preprocessing.py --dataset $i --thres $j --interpol GP";  
        #done;
    done;
done

