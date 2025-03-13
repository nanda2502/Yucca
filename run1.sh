#!/bin/bash
#SBATCH -p genoa
#SBATCH -n 1
#SBATCH --cpus-per-task 192
#SBATCH -t 12:00:00

export OMP_NUM_THREADS=192

cd build

./yucca 0 121
