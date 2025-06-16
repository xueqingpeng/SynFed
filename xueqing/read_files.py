import json


def read_jsonl_files(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)


if __name__ == "__main__":
    file_path = "/gpfs/radev/home/xp83/Documents/project/scripts/SynFed/syned_datasets/syn1_score/switzerland.score.jsonl"
    data = read_jsonl_files(file_path)
    for item in data:
        if item["status"] == "SUCCESS":
            reward_ds0 = item["reward_ds0"]
            for i in reward_ds0:
                print(i["question"])
                print(i["gold"])
                print(i["answer"])
                print("-"*100)
            break