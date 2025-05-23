#!/bin/bash
#SBATCH -p rome
#SBATCH -n 63
#SBATCH --cpus-per-task 25
#SBATCH -t 00:45:00

export OMP_NUM_THREADS=25

cd build

for i in {0..62}; do
    if [ ! -f "../output/expected_steps_${i}.csv.gz" ]; then
        ./yucca "$i" 6&
    fi
done

wait
cd ..
./combine.sh