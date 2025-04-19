#!/bin/bash

#SBATCH --job-name=synfed
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint="h100"
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xp83/Documents/project/logs/%j_gpu.out

module load miniconda
conda activate /gpfs/radev/home/xp83/project/env/synfed 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo '-------------------------------------------------'
echo "Job Name: ${SLURM_JOB_NAME}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s) on partition ${SLURM_JOB_PARTITION}"
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------------------------'
echo -e '\n\n'

# bash r0_s1_gen_synsft.sh
# bash r0_s3_gen_scoreanchors.sh

# bash r1_s1_syn_addprompt.sh
# python r1_s1.5_balance.py
bash r1_s2_raw_to_csv.sh
bash r1_s3_gen_sftfed_train.sh

# bash r1_s4_addreward.sh
