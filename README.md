# NUS_dsa5208_project3_MPI_OpenMP
For NUS DSA5208 Project 3 submission. Before proceeding, unzip all of the zipped and 7z files before continuing.

## Essential files:

Source code for logistic regression:
- lr_longdouble.cpp - CPU implementation
- lr_GPU_longdouble.cpp - GPU implementation
-- include folder includes the external packages used for this executable

Batch files:
- queue_compilation.sh - to compile the executable on HPC
-- compilation_cpu.sh - CPU compilation
-- compilation_gpu.sh - GPU compilation

- batch_run.sh - to batch qsub the jobs
-- execute_*.sh - CPU supporting sh 
-- execute_GPU_*.sh - GPU supporting sh 

Compiled executable:
- logistic_newton - CPU implementation
- logistic_newton_GPU - GPU implementation
Both executables were compiled on Vanda server

Dataset files
- processed_1.csv - log(rows)=13.65
- processed_2.csv - log(rows)=14.69
- processed.csv - log(rows)=15.08, training set for model analysis
- processed_4.csv - log(rows)=15.78
- processed_5.csv - log(rows)=16.47
- test.csv - test set

Python scripts:
- scikit-implementation.py - to obtain the results from sci-kit implementation
- data_processing.py - data cleanup for graph generation and statistical analysis


## 1) Compilation
The executables had been pre-compiled on Vanda server. If interested to recompile however,
excute the bash code in HPC. It will excute qsub for compilation_cpu.sh and compilation_gpu.sh

```
bash queue_compilation.sh
```


## 2) Execution

```
bash batch_run.sh
```

to execute the qsub jobs present in the report. If you prefer to modify the parameters for the executable,
the following CLI arguments are avaiable.

```
logistic_newton mode= iter=1000 l2=1 tol=1e-4 path="./processed.csv" output_folder="./output"
logistic_newton_GPU mode= iter=1000 l2=1 tol=1e-4 path="./processed.csv" output_folder="./output_gpu"
```

Arguements with values were default values. The argument description were as follows:

## Arguments

```mode=("training", "fit")```

- Training to start the training of the model, fit to predict the labels
- If ‘fit’ is set, the rest of the arguments were ignored
- Compulsory to input the keywords

```iter=Integer```	

- Maximum number of iterations

```l2=Float```

- L2 regularisation strength

```tol=Float```

- Tolerance, threshold for convergence

```path=string```

- Path to the excel csv

```output_folder=string```

- Path to output folder

