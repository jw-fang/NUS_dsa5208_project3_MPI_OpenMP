#!/bin/bash
# qsub -q cpu_parallel -l select=1:ncpus=8 -j oe -o $PWD execute.sh
# -q serial: min/max cpu=1; parallel min 12 max 96
cd $PBS_O_WORKDIR; 
module load openmpi
module load cuda
export OMP_PLACES=cores
export OMP_PROC_BIND=close
MPI_RANKS=$(wc -l < "$PBS_NODEFILE")

echo "MPI_ranks ${MPI_RANKS},openmpi ${OMP_NUM_THREADS}" 
for ((i=0;i<10;i++)); do
    mpirun --bind-to core --map-by core ./logistic_newton_GPU mode=training
done

