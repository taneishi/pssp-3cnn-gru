#!/bin/sh
#PBS -o log/$PBS_JOBID.out
#PBS -j oe

hostname | grep '^s'
lscpu | grep 'Model name\\|^CPU(s)'

if [ $PBS_O_WORKDIR ]; then cd $PBS_O_WORKDIR; mkdir -p log; fi

if [ -d /opt/intel/inteloneapi ]; then 
    source /opt/intel/inteloneapi/setvars.sh
    conda activate pytorch
fi

python main.py
