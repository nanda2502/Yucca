#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <num_nodes>"
    exit 1
fi


num_nodes="$1"


case "$num_nodes" in
    8)
        max_index=2044
        ;;
    7)
        max_index=317
        ;;
    6)
        max_index=63
        ;;
    50)
        max_index=48
        ;;
    *)
        echo "Invalid num_nodes. Please provide either 7 or 8."
        exit 1
        ;;
esac

# Initialize counter for missing files
missing_count=0

# Loop through expected files and count missing ones
for i in $(seq 0 $max_index); do
    if [ ! -f "./output/expected_steps_${i}.csv.gz" ]; then
        echo "Missing file: expected_steps_${i}.csv.gz"
        missing_count=$((missing_count + 1))
    fi
done

# Report the number of missing files
echo $missing_count