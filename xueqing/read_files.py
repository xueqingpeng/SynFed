import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_jsonl_files(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)


def draw_histogram(data, output_path):
    plt.figure()
    plt.hist(data, bins=100, color="skyblue", edgecolor="black")
    plt.title("Histogram of Gain Scores")
    plt.xlabel("Gain Score (Status Score == 5)")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


def main():
    round = "syn1"
    data_names = ["cleveland", "hungarian", "switzerland"]
    for data_name in data_names:
        print(f"Processing {data_name}...")
        data_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_score/{data_name}.score.jsonl"
        data = read_jsonl_files(data_fp)

        # # Draw histogram of gain scores
        # png_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/xueqing/{round}_{data_name}.score.png"
        # gain_scores = []
        # for item in data:
        #     if item["status"] == "SUCCESS":
        #         gain_scores.append(item["gain_score"])
        #     #     reward_ds0 = item["reward_ds0"]
        #     #     for i in reward_ds0:
        #     #         print(i["question"])
        #     #         print(i["gold"])
        #     #         print(i["answer"])
        #     #         print("-"*100)
        #         # break
        # draw_histogram(gain_scores, png_fp)

        # Filter out scores
        df = pd.DataFrame(list(data))[["status", "gain_score", "status_score"]]

        df_success = df[df["status"] == "SUCCESS"]
        print("100% quantile:", df_success["gain_score"].quantile(1.0))
        print("80% quantile:", df_success["gain_score"].quantile(0.8))
        print("20% quantile:", df_success["gain_score"].quantile(0.2))
        print("0% quantile:", df_success["gain_score"].quantile(0.0))
        
        df_top = df_success.nlargest(int(len(df_success) * 0.2), 'gain_score')
        df_bottom = df_success.nsmallest(int(len(df_success) * 0.2), 'gain_score')
        print(f"Top 20% gain scores range: {df_top['gain_score'].min()} - {df_top['gain_score'].max()}. ({len(df_top)}/{len(df_success)} rows selected.)")
        print(f"Bottom 20% gain scores range: {df_bottom['gain_score'].min()} - {df_bottom['gain_score'].max()}. ({len(df_bottom)}/{len(df_success)} rows selected.)")
        df_positive = df_success[df_success['gain_score'] > 0]
        df_negative = df_success[df_success['gain_score'] < 0]
        print(f"Positive gain scores range: {df_positive['gain_score'].min()} - {df_positive['gain_score'].max()}. ({len(df_positive)}/{len(df_success)} rows selected.)")
        print(f"Negative gain scores range: {df_negative['gain_score'].min()} - {df_negative['gain_score'].max()}. ({len(df_negative)}/{len(df_success)} rows selected.)")

        fp_sft = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_sft_fed/{data_name}_client_0.train.jsonl"
        fp_sft_top = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_sft_fed/{data_name}_client_0_top_20.train.jsonl"
        fp_sft_bottom = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_sft_fed/{data_name}_client_0_bottom_20.train.jsonl"
        fp_sft_positive = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_sft_fed/{data_name}_client_0_positive.train.jsonl"
        fp_sft_negative = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/{round}_sft_fed/{data_name}_client_0_negative.train.jsonl"

        data_sft = read_jsonl_files(fp_sft)
        df_sft = pd.DataFrame(list(data_sft))
        df_sft.iloc[df_top.index,:].to_json(fp_sft_top, orient="records", lines=True)
        df_sft.iloc[df_bottom.index,:].to_json(fp_sft_bottom, orient="records", lines=True)
        df_sft.iloc[df_positive.index,:].to_json(fp_sft_positive, orient="records", lines=True)
        df_sft.iloc[df_negative.index,:].to_json(fp_sft_negative, orient="records", lines=True)

        print(f"Done {data_name}!")


if __name__ == "__main__":
    main()









# Processing cleveland...
# 100% quantile: 0.5163977794943223
# 80% quantile: 0.0
# 20% quantile: -0.5163977794943222
# 0% quantile: -0.8355491589367869
# Top 20% gain scores range: 0.0 - 0.5163977794943223. (1215/6078 rows selected.)
# Bottom 20% gain scores range: -0.8355491589367869 - -0.5163977794943222. (1215/6078 rows selected.)
# Positive gain scores range: 0.1197655832620661 - 0.5163977794943223. (1069/6078 rows selected.)
# Negative gain scores range: -0.8355491589367869 - -0.2581988897471611. (4371/6078 rows selected.)
# Done cleveland!
# Processing hungarian...
# 100% quantile: 0.19724640005185756
# 80% quantile: -0.31915137944246474
# 20% quantile: -0.5773502691896258
# 0% quantile: -0.8355491589367869
# Top 20% gain scores range: -0.31915137944246474 - 0.19724640005185756. (80/404 rows selected.)
# Bottom 20% gain scores range: -0.8355491589367869 - -0.5773502691896258. (80/404 rows selected.)
# Positive gain scores range: 0.19724640005185756 - 0.19724640005185756. (6/404 rows selected.)
# Negative gain scores range: -0.8355491589367869 - -0.07735026918962584. (371/404 rows selected.)
# Done hungarian!
# Processing switzerland...
# 100% quantile: 0.8355491589367869
# 80% quantile: 0.5163977794943222
# 20% quantile: 0.2581988897471611
# 0% quantile: -0.31915137944246474
# Top 20% gain scores range: 0.5163977794943222 - 0.8355491589367869. (169/847 rows selected.)
# Bottom 20% gain scores range: -0.31915137944246474 - 0.2581988897471611. (169/847 rows selected.)
# Positive gain scores range: 0.2581988897471611 - 0.8355491589367869. (684/847 rows selected.)
# Negative gain scores range: -0.31915137944246474 - -0.1197655832620661. (46/847 rows selected.)
# Done switzerland!