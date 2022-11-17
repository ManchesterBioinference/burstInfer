#!/bin/bash --login
# -- SGE options
#$ -cwd
#$ -pe smp.pe 2
#$ -t 1-50

module load apps/binapps/anaconda3/2019.03
module load tools/env/proxy

export OMP_NUM_THREADS=2

python uwt_e1_no_bd_centre.py
#-out results.$SGE_TASK_ID.csv
