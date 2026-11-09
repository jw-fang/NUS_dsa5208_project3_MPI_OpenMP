#!/bin/bash
# qsub -q cpu_parallel -l select=1:ncpus=8 -j oe -o $PWD execute.sh
# -q serial: min/max cpu=1; parallel min 12 max 96
cd $PBS_O_WORKDIR; 
export LD_LIBRARY_PATH=/opt/pbs/lib:$LD_LIBRARY_PATH
module load openmpi
export OMP_PLACES=cores
export OMP_PROC_BIND=close
MPI_RANKS=$(wc -l < "$PBS_NODEFILE")

sort "$PBS_NODEFILE" | uniq -c |
awk '{print $2 " slots=" $1}' > mpi_hosts
echo "MPI_ranks ${MPI_RANKS},openmpi ${OMP_NUM_THREADS}" 
for ((i=0;i<10;i++)); do
    mpirun -np "$MPI_RANKS" --bind-to core --map-by core ./logistic_newton mode=training path=./processed_2.csv output_folder=./processed_2_MPI
done

