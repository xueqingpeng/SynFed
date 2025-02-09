#!/bin/bash
set -x  # Enable debugging

# Change to the working directory
cd "$work_dir" || exit 1  # Exit if the directory change fails

# Define dataset directory and dataset list
dataset_dir="syned_datasets/syn1_dpo"
# datasets=("sft_syn_abalone" "sft_syn_adult" "sft_syn_buddy" "sft_syn_california" "sft_syn_diabetes" "sft_syn_insurance" "sft_syn_german")
# datasets=("sft_syn_abalone")
datasets=("diabetes")
pretrain="syn_checkpoints/llama32-3b-sft_syn_diabetes"

# Loop through each dataset
for ds_name in "${datasets[@]}"; do
  echo "Running training for dataset: $ds_name"

  # Define the training command
  read -r -d '' training_commands <<EOF
  openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-3b-dpo-$ds_name \
   --pretrain $pretrain \
   --dataset $dataset_dir/$ds_name \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 192 \
   --micro_train_batch_size 8 \
   --bf16 \
   --max_epochs 2 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --chosen_key chosen \
   --rejected_key reject \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb 7fa9b84bacf616286628d94b0b45ffde280278e1 \
   --wandb_run_name "dpo_${ds_name}_$(date +%Y%m%d_%H%M%S)" \
   --apply_chat_template \

EOF

  # Run the command with deepspeed
  deepspeed --include=localhost:1,2,3,4,5,6 --module $training_commands
  
  # Check if the job failed
  if [ $? -ne 0 ]; then
    echo "Job for $ds_name failed. Moving to the next dataset."
  else
    echo "Job for $ds_name completed successfully."
  fi
done

echo "All jobs have completed."