#!/bin/sh

module load python/3.6.0
pipenv shell

index=$1

#python src/main.py --dataset $index

for i in $(seq 0 $index); do 
   for j in $(seq 0 20 80); do 
        #bsub "./test_script.sh $i $j" -n 5 -R "rusage[mem=4096]"; 
        #if [ ! -e outfiles/total_dtw_distances_channel_${j}_horizon_${i}_vs_horizon_0.npz ]; then
        echo submitting job $i, $j 
        bsub -n 2 -W 8:00 -R "rusage[mem=2048]" -e "jobs/" -o "jobs/" "python src/main.py --dataset $i --thres $j";  
        #fi
    done;
done

