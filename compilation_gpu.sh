#!/bin/bash
#PBS -S /bin/bash
cd "$PBS_O_WORKDIR"
module load nvhpc
module load openmpi

nvidia-smi
which nvc++


nvc++ -O0 -std=c++17 -mp=gpu -gpu=nofma     -DEIGEN_DONT_VECTORIZE     -Iinclude -Iinclude/eigen3     $(mpic++ --showme:compile)     ./lr_GPU_longdouble.cpp     $(mpic++ --showme:link)     -o logistic_newton_GPU
