#!/bin/bash

#SBATCH --job-name=preproc

# Define how long you job will run d-hh:mm:ss
#SBATCH --time 3-00:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G


cd ..

python ./preprocess_cv.py
