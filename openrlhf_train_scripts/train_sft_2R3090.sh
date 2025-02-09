set -x
# Change to the working directory
cd "$work_dir" || exit 1  # Exit if the directory change fails

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset ./sft_dataset/train \
   --input_key question \
   --output_key response \
   --train_batch_size 16 \
   --micro_train_batch_size 8 \
   --max_samples 500000 \
   --pretrain /data/wangyuxin/rayzhang/MODELS/meta-llama--Llama-3.2-1B-Instruct \
   --save_path ./checkpoint/llama32-1b-sft \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 3e-6 \
   --load_checkpoint \
   --use_wandb 7fa9b84bacf616286628d94b0b45ffde280278e1 \
   --gradient_checkpointing \
   --adam_offload 

EOF
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi