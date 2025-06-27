import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import matthews_corrcoef
from datasets import load_dataset


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="balanced",  # Automatically balance across available devices
        torch_dtype=torch.bfloat16,  # 或使用 float16/float32，视你的硬件而定
        trust_remote_code=True
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="balanced")
    return pipe


def run_evaluation(data, prompt_key, gold_key, pipe):
    results = []
    for item in tqdm(data, desc="Running evaluation"):
        prompt = item[prompt_key]
        gold = item[gold_key]

        output = pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        generated = output[len(prompt):].strip()

        results.append({
            "prompt": prompt,
            "gold": gold,
            "generated": generated,
            "pred": "unhealthy" if "unhealthy" in generated else "healthy"
        })
    return results


def calculate_mcc(results):
    label_map = {
        "unhealthy": 0,
        "healthy": 1
    }

    y_true = []
    y_pred = []
    for r in results:
        true = r["gold"].strip().lower()
        pred = r["pred"].strip().lower()

        if true in label_map and pred in label_map:
            y_true.append(label_map[true])
            y_pred.append(label_map[pred])

    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


model_paths = [
    # "TheFinAI/fl-cleveland-sft-0",
    # "TheFinAI/fl-cleveland-sft-1-adapter",
    # "TheFinAI/fl-hungarian-sft-1-adapter",
    # "TheFinAI/fl-switzerland-sft-1-adapter",
    "TheFinAI/fl-magnitude_prune-1-sft-merged-base-62",
    "TheFinAI/fl-cleveland-sft-1-adapter",
    "TheFinAI/fl-hungarian-sft-1-adapter",
    "TheFinAI/fl-switzerland-sft-1-adapter"
]
data_paths = {
    "cleveland": "TheFinAI/MED_SYN0_CLEVELAND_test",
    "hungarian": "TheFinAI/MED_SYN0_HUNGARIAN_test",
    "switzerland": "TheFinAI/MED_SYN0_SWITZERLAND_test",
}
data_paths = {
    "cleveland": "cleveland_anchors.json",
    "hungarian": "hungarian_anchors.json",
    "switzerland": "switzerland_anchors.json",
}
mccs = []
for model_path in model_paths:
    print(f"Evaluating {model_path}")
    pipe = load_model(model_path)
    for data_name, data_path in data_paths.items():
        print(f"Evaluating {data_name}")
        # dataset = load_dataset(data_path)
        # train_data = dataset["train"]
        anchors = json.load(open(data_path))

        # Evaluate on train data
        # results = run_evaluation(train_data, prompt_key="query", gold_key="answer", pipe=pipe)
        results = run_evaluation(anchors, prompt_key="question", gold_key="gold", pipe=pipe)
        # with open(f"train_results_{data_name}.json", "w") as f:
        with open(f"anchors_results_{data_name}.json", "w") as f:
            json.dump(results, f)
        mcc = calculate_mcc(results)
        print(f"MCC for {data_name} test: {mcc:.4f}")
    

    # anchors

    # zero_shot_results = {}
    # few_shot_results = {}
    # model_results = {
    #     model_name: {
    #         "zero-shot": zero_shot_results,
    #         "few-shot": few_shot_results
    #     }
    # }
    # mccs.append(model_results)

    # for data_name in data_names:
    #     data = test_zero_shot_all[data_name]
    #     mcc_zero_shot = calculate_mcc(run_evaluation(data, prompt_key="question", gold_key="gold", pipe))
    #     print(f"MCC for {data_name} zero-shot: {mcc_zero_shot:.4f}")
    #     zero_shot_results[data_name] = mcc_zero_shot

    # for data_name in data_names:
    #     data = test_few_shot_all[data_name]
    #     mcc_few_shot = calculate_mcc(run_evaluation(data, prompt_key="question", gold_key="gold", pipe))
    #     print(f"MCC for {data_name} few-shot: {mcc_few_shot:.4f}")
    #     few_shot_results[data_name] = mcc_few_shot

# with open("mccs.json", "w") as f:
#     json.dump(mccs, f)
