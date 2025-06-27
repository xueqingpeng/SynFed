import json
import matplotlib.pyplot as plt


def read_jsonl_files(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)


def draw_histogram(data, output_path):
    plt.hist(data, bins=100, color="skyblue", edgecolor="black")
    plt.title("Histogram of Gain Scores")
    plt.xlabel("Gain Score (Status Score == 5)")
    plt.ylabel("Frequency")
    plt.savefig(output_path)


if __name__ == "__main__":
    data_names = ["cleveland", "hungarian", "switzerland"]
    for data_name in data_names:
        data_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_score/{data_name}.score.jsonl"
        png_fp = f"/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/xueqing/{data_name}.score.png"
        print(f"Processing {data_name}...")

        data = read_jsonl_files(data_fp)
        gain_scores = []
        for item in data:
            if item["status"] == "SUCCESS":
                gain_scores.append(item["gain_score"])
            #     reward_ds0 = item["reward_ds0"]
            #     for i in reward_ds0:
            #         print(i["question"])
            #         print(i["gold"])
            #         print(i["answer"])
            #         print("-"*100)
                # break
        draw_histogram(gain_scores, png_fp)

        print(f"Done {data_name}!")