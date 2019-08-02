#!/bin/bash
#
#SBATCH --job-name=eeg_multiclass
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=128G


code_dir='/home/tsy935/RubinLab_neurotranslate_eeg/eeg_reports'

cd ${code_dir}


srun hostname
srun python3 ${code_dir}/EEG_Doc_MultiClassClassification_Metal_v0.5.0.py

exit 0

