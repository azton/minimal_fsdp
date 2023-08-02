#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q R564516
#PBS -A tpc
#PBS -N minfsdp

# Controlling the output of your application
# UG Sec 3.3 page UG-40 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error

cd $PBS_O_WORKDIR
DIR=$PBS_O_WORKDIR
module load conda/2023-01-10-unstable 
conda activate /lus/eagle/projects/tpc/azton/fsdp_pt2.0
echo PYTHON:
python --version

echo PYTORCH:

python <<< "import torch; print(torch.__version__)"
# conda activate /home/azton/pytorch-2.0
# Internet access on nodes
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3130
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128
echo "Set HTTP_PROXY and to $HTTP_PROXY"

# Set ADDR and PORT for communication
master_node=$(cat $PBS_NODEFILE| head -1)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')
echo "MASTER NODE ${master_node} :: MASTER_ADDR ${MASTER_ADDR}"
export MASTER_PORT=23450

#cuda tuning
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MPICH_GPU_SUPPORT_ENABLED=1
# MPI and OpenMP settings
export NNODES=`wc -l < $PBS_NODEFILE`
export NRANKS_PER_NODE=4

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo < $PBS_NODEFILE
echo "TASK NUM = ${PBS_TASKNUM}"

ulimit -s unlimited

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

mpiexec -n ${NTOTRANKS} \
        -ppn $NRANKS_PER_NODE \
        --cpu-bind verbose,list:0,16,32,48 \
        --hostfile $PBS_NODEFILE python -u basic_training.py \
                --max_epochs 20 \
                --environment pbs >> basic_training.out 2>&1
