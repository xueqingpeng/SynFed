#!/bin/bash
set -x  # Enable debugging

# Change to the working directory
cd "$work_dir" || exit 1  # Exit if the directory change fails

# Define dataset directory and dataset list
dataset_dir="./syn_datasets"
# datasets=("sft_syn_abalone" "sft_syn_adult" "sft_syn_buddy" "sft_syn_california" "sft_syn_diabetes" "sft_syn_insurance" "sft_syn_german")
# datasets=("sft_syn_abalone")
datasets=("sft_syn_german")


# Loop through each dataset
for ds_name in "${datasets[@]}"; do
  echo "Running training for dataset: $ds_name"

  # Define the training command
  read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --ckpt_path ckpt/$ds_name \
   --max_ckpt_num 5 \
   --max_len 2048 \
   --dataset $dataset_dir/$ds_name \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 16 \
   --max_samples 500000 \
   --pretrain /data/wangyuxin/rayzhang/pretrain/meta-llama--Llama-3.2-3B-Instruct \
   --save_path ./checkpoint/llama32-3b-$ds_name \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-6 \
   --load_checkpoint \
   --use_wandb 7fa9b84bacf616286628d94b0b45ffde280278e1 \
   --gradient_checkpointing \
   --adam_offload \
   --wandb_run_name "${ds_name}_$(date +%Y%m%d_%H%M%S)"
EOF

  # Run the command with deepspeed
  deepspeed --include=localhost:0,1,2,3,4,5,6,7 --module $training_commands
  
  # Check if the job failed
  if [ $? -ne 0 ]; then
    echo "Job for $ds_name failed. Moving to the next dataset."
  else
    echo "Job for $ds_name completed successfully."
  fi
done

echo "All jobs have completed."
