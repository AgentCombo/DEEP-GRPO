import os
import json

import datasets

def format_prompt(question, constraints, format, file_name):
    prompt = f"Question: {question}\nConstraints: {constraints}\nFormat: {format}\nFile Name: {file_name}"
    return prompt

DATA_SOURCE = "DABench"
DATA_PATH = "/data/InfiAgent/examples/DA-Agent/data"
WORK_SPACE = "/data/InfiAgent/examples/DA-Agent/data/da-dev-tables"
SAVE_PATH = "/data/hf-datasets/dabench"

with open(os.path.join(DATA_PATH, "da-dev-questions.jsonl"), "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f.readlines()]

with open(os.path.join(DATA_PATH, "da-dev-labels.jsonl"), "r", encoding="utf-8") as f:
    labels = [json.loads(line) for line in f.readlines()]

assert len(questions) == len(labels)

for question, label in zip(questions, labels):
    assert question["id"] == label["id"]
    question["label"] = label["common_answers"]

dataset = datasets.Dataset.from_list(questions)

def make_map_fn(split):
    def process_fn(example):
        id = example.pop("id")
        question = example.pop("question")
        concepts = example.pop("concepts")
        constraints = example.pop("constraints")
        format = example.pop("format")
        file_name = example.pop("file_name")
        level = example.pop("level")
        label = example.pop("label")

        data = {
            "data_source": DATA_SOURCE,
            "prompt": [
                {
                    "role": "user",
                    "content": format_prompt(question, constraints, format, file_name),
                }
            ],
            "ability": "data_analysis",
            "reward_model": {"style": "rule", "ground_truth": label},
            "agent_name": "deep_analyze_agent_loop",
            "extra_info": {
                "split": split,
                "index": id,
                "question": question,
                "concepts": concepts,
                "constraints": constraints,
                "format": format,
                "init_files": [file_name],
                "level": level,
                "label": label,
                "workspace": WORK_SPACE
            },
        }
        return data

    return process_fn

val_dataset = dataset.map(function=make_map_fn("val"))

os.makedirs(SAVE_PATH, exist_ok=True)
val_dataset.to_parquet(os.path.join(SAVE_PATH, "val.parquet"))