#!/bin/bash

qsub -j oe -o "$PWD" -l select=1:ncpus=1:ngpus=1  -l walltime=00:30:00 compilation_gpu.sh
qsub -j oe -o "$PWD" -l select=1:ncpus=1 -l walltime=00:30:00 compilation_cpu.sh


