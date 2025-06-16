#!/bin/bash

#SBATCH --job-name=synfed
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --constraint="h100"
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xp83/Documents/project/logs/%j_gpu.out

module load miniconda
conda activate /gpfs/radev/home/xp83/project/env/synfed 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo '---------------------------------------------------------------------------'
echo "Job Name              : ${SLURM_JOB_NAME}"
echo "Running on Host       : $(hostname)"
echo "Partition             : ${SLURM_JOB_PARTITION}"
echo "CPUs on Node          : ${SLURM_CPUS_ON_NODE}"
echo "GPUs on Node          : ${SLURM_GPUS_ON_NODE:-0}"
echo "CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES:-None}"
echo "Environment Name      : ${CONDA_DEFAULT_ENV:-Not Set}"
echo "Node List             : ${SLURM_NODELIST}"
echo "Submit Directory      : ${SLURM_SUBMIT_DIR}"
echo "Start Time            : $(date)"
echo '---------------------------------------------------------------------------'
echo -e '\n\n'

export HF_HOME=/gpfs/radev/scratch/xu_hua/shared/hf_models



source .env
echo "HF_TOKEN: $HF_TOKEN" | cut -c1-20

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export NCCL_P2P_DISABLE=1
export HYDRA_FULL_ERROR=1
export N_CUDA=2

# bash r0_s1_gen_synsft.sh
# bash r0_s3_gen_scoreanchors.sh

# bash r1_s1_syn_addprompt.sh
# python r1_s1.5_balance.py
# bash r1_s2_raw_to_csv.sh
# bash r1_s3_gen_sftfed_train.sh

# bash r1_s4_addreward.sh
# bash r1_s5_gen_dpo_train.sh
