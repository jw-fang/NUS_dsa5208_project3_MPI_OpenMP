#!/bin/bash
#PBS -S /bin/bash
cd "$PBS_O_WORKDIR"
module load openmpi
which mpic++

mpic++ -g -O0 -std=c++17 -fopenmp -pthread -Iinclude -Iinclude/eigen3 -ffp-contract=off lr_longdouble.cpp -o logistic_newton
