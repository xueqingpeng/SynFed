set -x
# Change to the working directory
cd "$work_dir" || exit 1  # Exit if the directory change fails

dataset_dir=/data/wangyuxin/rayzhang/synllm/syn_datasets
# dataset=(sft_syn_abalone  sft_syn_adult  sft_syn_buddy  sft_syn_california  sft_syn_diabetes  sft_syn_insurance)
ds_name=sft_syn_abalone

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --ckpt_path ckpt/$ds_name \
   --max_ckpt_num 5 \
   --max_len 2048 \
   --dataset $dataset_dir/$ds_name \
   --input_key question \
   --output_key response \
   --train_batch_size 192 \
   --micro_train_batch_size 24 \
   --max_samples 500000 \
   --pretrain /data/wangyuxin/rayzhang/pretrain/meta-llama--Llama-3.2-1B-Instruct \
   --save_path ./checkpoint/llama32-1b-$ds_name \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 4 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-6 \
   --load_checkpoint \
   --use_wandb 7fa9b84bacf616286628d94b0b45ffde280278e1 \
   --gradient_checkpointing \
   --adam_offload \
   --wandb_run_name "${ds_name}_$(date +%Y%m%d_%H%M%S)"

EOF
    # --packing_samples


deepspeed  --include=localhost:4,5,6,7 --module $training_commands
