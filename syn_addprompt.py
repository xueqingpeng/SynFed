import os

import hydra
from omegaconf import DictConfig

from datasets import load_from_disk
from src.client import SynClient, TextToDictStatus
from src.data import SCHEMA_MAP
from src.logger import get_logger
from src.rewarder import AddonePromptAdder, UnsimilarDataFinder
from src.utils import (generate_uuid, mask_safe_json_serializable,
                       write_jsonlines)

logger = get_logger(__name__)


def synthesize_data(cfg, syn_client, prompt_data):
    """Generates synthetic data for the prompt dataset."""
    results = []
    prompts = []
    
    num_repeat = cfg.num_repeat
    
    for i, data in enumerate(prompt_data):
        syn_pmt_id = generate_uuid()
        results.extend([{'syn_pmt_id':syn_pmt_id, "syn_id":generate_uuid()} for i in range(num_repeat)])
        prompts.extend([data['question']] * num_repeat)
        
        if cfg.debug:
            if i > 5:
                break
            
    syn_datas = syn_client.generate(prompts)
    for r, s in zip(results, syn_datas):
        r.update(s)
        
    return results

def add_prompt_data(results, adder):
    """Adds prompt data to the generated synthetic data."""
    for data in results:
        if data.get('status') is TextToDictStatus.SUCCESS:
            dpoint = data['result']
            # Add prompt data
            prompts = adder(dpoint)
            data.update(prompts)
    return results

def safe_write_to_disk(results, save_path):
    for result in results:
        result['status'] = str(result['status'])
        result['result'] = mask_safe_json_serializable(result['result'])
    write_jsonlines(results, save_path)
    return 

@hydra.main(config_path="./", config_name="syn_addprompt", version_base="1.3")
def main(cfg: DictConfig):
    # Retrieve the specific dataset configuration
    ds_name = cfg.ds_name
    if ds_name not in cfg.datasets:
        raise ValueError(f"Dataset '{ds_name}' not found in the configuration.")

    ds_config = cfg.datasets[ds_name]
    logger.info("*" * 50 + f"  Processing dataset: {ds_name}\n\n")

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"{ds_name}.raw.jsonl")

    # Load schema and data
    schema = SCHEMA_MAP.get(ds_name)
    prompt_data = load_from_disk(ds_config.prompt_data)['test']
    anchor_data = schema.format_load_csv(ds_config.anchor_data)
    train_data = schema.format_load_csv(ds_config.train_data)

    syn_config = cfg.syn_config
    syn_config.vllm_params.model = ds_config.syn_model

    # 1. Generate synthetic data
    logger.info("+" * 30 + "  Generating synthetic data...")
    with SynClient(syn_config, schema) as syn_client:
        results = synthesize_data(cfg, syn_client, prompt_data)


    # Find Unsimilar prompt data for scoring
    data_finder = UnsimilarDataFinder(schema, train_data, 4)
    unsim_anchor_data = []
    for anchor in anchor_data:
        unsim_anchor_data.append(data_finder.get_not_sames(anchor))

    # 2. Add prompt data
    adder = AddonePromptAdder(
        schema=schema,
        anchor_data=anchor_data,
        unsim_anchor_data=unsim_anchor_data,
    )
    results = add_prompt_data(results, adder)

    safe_write_to_disk(results, save_path)



if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
