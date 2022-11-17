#!/bin/bash --login
# -- SGE options
#$ -cwd
#$ -pe smp.pe 4
#$ -t 1-50

module load apps/binapps/anaconda3/2019.03
module load tools/env/proxy

export OMP_NUM_THREADS=4

python uwt_e1_no_bd_outer.py
#-out results.$SGE_TASK_ID.csv
