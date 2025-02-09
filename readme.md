## **Script Naming Convention Explanation**

The script naming convention follows this format:

**r{x}_s{x}_{description}**
- `r{x}`: Round number (iteration stage)
- `s{x}`: Step number (specific process within the round)
- `{description}`: Brief explanation of the step

---

## **1. Cold Start (Round 0)**

### **r0_s1_gen_synsft.sh**
- Calls **`src.data.sft_generator`**, which generates the initial synthetic fine-tuning dataset.
- Uses datasets **A, B, and C** to generate training data, while dataset **D** is used to generate prompts.
- The output is stored in **`syn_datasets/`**.
- The training framework used is **OpenRLHF**, with training scripts available in **`openrlhf_train_scripts/train_sft_all.sh`**.

### **r0_s2_gen_sftfed_all.sh**
- Calls **`src.data.sft_federated`**, which creates datasets for downstream tasks.
- The output is stored in **`syned_datasets/original_0_sft_fed/`**.

### **r0_s3_gen_scoreanchors.sh**
- Calls **`src.data.synscore_anchors`**, which clusters and extracts category centers from the test set to reduce the scale of the scoring dataset.
- The output is stored in **`./score_anchors/`**.

---

## **2. First Iteration of Synthetic Data (Round 1)**

### **r1_s1_syn_addprompt.sh**
- Calls **`syn_addprompt.py`**, which performs three main tasks:
  1. **Synthetic Answer Generation (`syn`)**: Uses the test set from **r0_s1** to generate synthetic answers.
  2. **Answer Extraction (`extract`)**: Converts answers into a standardized format based on **`src.data.schema`**, which varies by dataset.
  3. **Prompt Augmentation (`addprompt`)**: Integrates scoring datasets from **r0_s3** into each synthetic data sample, storing them in `ds0` and `ds1` fields.

### **r1_s2_raw_to_csv.sh**
- Calls **`convert_raw_to_csv.py`**, which:
  - Converts the raw dataset from **r1_s1** into **CSV format**.
  - Removes unnecessary fields.

### **r1_s3_gen_sftfed_train.sh**
- Similar to **r0_s2**, but generates downstream task datasets **only for training**.
- Uses the **CSV data generated in r1_s2**.

### **r1_s4_addreward.sh**
- Calls **`syn_addscore.py`**, which:
  - Uses the `ds0` and `ds1` scoring datasets from **r1_s1**.
  - Assigns scores to each data sample and stores them in the **`score` field**.

### **r1_s5_gen_dpo_train.sh**
- Calls **`convert_score_to_prefer.py`**, which:
  - **Ranks synthetic data** based on the `score` field.
  - Generates **preference pairs** for **DPO (Direct Preference Optimization) training**.
  - Outputs data in **OpenRLHF format**, with training scripts available in **`openrlhf_train_scripts/train_dpo.sh`**.

---

## **3. Key Components and Configuration**
- **`./src/data/synscore_anchors.yaml`**: The `num_clusters` parameter determines the size of the reward dataset. A larger value ensures better representation but increases computational cost.
- **`./syn_addprompt.yaml`**: The `num_repeat` parameter controls the number of synthetic prompts generated per data point. Increasing this value leads to a larger synthetic dataset. If set high, adjust `syn_config.sampling_params.temperature` and `top_p` accordingly.

---

## **4. Model Descriptions**
- **Synthetic Model training set (`syn_datasets`)**: Used to train the cold-start model. Ideally, each node should provide its own cold-start model. However, for consistency, a unified synthetic cold-start model is used, trained on `abalone`, `adult`, `buddy`, and `california` datasets, ensuring it has not seen `med` and `fin` datasets.
- **Synthetic Data Model Configuration (`./syn_addprompt.yaml`)**: Each dataset has its own configuration, set via `syn_model: "syn_checkpoints/llama32-3b-sft_syn_diabetes"`.
- **Reward Model**: The evaluation model used for reward scoring is defined in `syn_addscore.py` under `client_config`. This model should use a merged model and should be trained using `r1_s3` data to create the `scalebio` merge model.
