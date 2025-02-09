work_dir="$(dirname "$(realpath "$0")")"
echo $work_dir
export PYTHONPATH="$work_dir/lm_evaluation:$work_dir/lm_evaluation/financial-evaluation:$work_dir/lm_evaluation/metrics/BARTScore"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

pretrained_model=eval_checkpoints/llama3.2-3BI-original-lora-merged
max_model_len=2048
max_gen_toks=64

# tasks=("syn_cls_german" "syn_cls_adult" "syn_cls_diabetes" "syn_cls_buddy" "syn_reg_abalone" "syn_reg_california" "syn_reg_insurance")

# task="cls_german"
task="cls_adult"
# task="cls_diabetes"

python lm_evaluation/eval.py \
    --model hf-causal-vllm \
    --tasks $task \
    --model_args use_accelerate=True,pretrained=$pretrained_model,tokenizer=$pretrained_model,max_model_len=$max_model_len,use_fast=False,dtype=float16 \
    --no_cache \
    --batch_size 10000 \
    --write_out 