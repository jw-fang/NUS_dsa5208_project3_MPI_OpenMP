#!/bin/bash

# paths=("./processed_1.csv" "./processed_2.csv" "./processed.csv")
# serial
for i in {1..5}; do
    qsub -j oe -o "$PWD" \
		-v OMP_NUM_THREADS=1 \
        -l walltime=1:00:00 \
        -l select=1:ncpus=1:mpiprocs=1:ompthreads=1 \
        "execute_${i}.sh"
done

# MPI only
nodes=(1 2 4 6 8)
for i in {1..5}; do
	for node in "${nodes[@]}"; do
    qsub -j oe -o "$PWD" \
		-v OMP_NUM_THREADS=1 \
        -l walltime=2:00:00 \
        -l select=$node:ncpus=32:mpiprocs=32:ompthreads=1 \
        "execute_${i}.sh"
	done
done

# GPU
cores=(1 2 4 9 18 36)
threads=(36 18 9 4 2 1)

for x in {1..5}; do
	for i in "${!cores[@]}"; do
		qsub -j oe -o "$PWD" \
			-v OMP_NUM_THREADS=${threads[i]} \
			-l walltime=00:30:00 \
			-l select=2:ncpus=36:mpiprocs=${cores[i]}:ompthreads=${threads[i]}:ngpus=1  \
			"execute_gpu_${x}.sh"
	done
done

for x in {1..5}; do
	for i in "${!cores[@]}"; do
		qsub -j oe -o "$PWD" \
			-v OMP_NUM_THREADS=1 \
			-l walltime=00:30:00 \
			-l select=2:ncpus=36:mpiprocs=${cores[i]}:ompthreads=1:ngpus=1  \
			"execute_gpu_${x}.sh"
	done
done

