#!/bin/bash
#SBATCH -p genoa
#SBATCH --array=1-7
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=3     # Only 3 tasks per node
#SBATCH --cpus-per-task=12      # 12 CPUs per task (36 of 192 cores per node)
#SBATCH --mem=330G              # Almost all available memory per node (336G)
#SBATCH -t 00:45:00
#SBATCH --output=slurm-%A.out   # Avoid scattering the output files

export OMP_NUM_THREADS=12

cd build

total_tasks=49  # 0 to 48
tasks_per_group=7  # 49/7 rounded
start=$(((SLURM_ARRAY_TASK_ID - 1) * tasks_per_group ))
end=$(( SLURM_ARRAY_TASK_ID * tasks_per_group - 1))

# Ensure we don't exceed total tasks
if [ $end -gt 48 ]; then
    end=48
fi

# Create a file to track running processes
running_pids="/tmp/running_pids_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $running_pids

echo "Starting tasks $start to $end on node $SLURM_NODEID"

# Run one task at a time to prevent memory issues
for i in $(seq $start $end); do
    if [ ! -f "../output/expected_steps_${i}.csv.gz" ]; then
        echo "Starting task $i"
        ./yucca "$i" 50
        echo "Completed task $i"
    else
        echo "Skipping task $i (output file already exists)"
    fi
done

# Clean up
rm $running_pids