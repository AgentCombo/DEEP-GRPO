import os

from tqdm import tqdm
import json
from datasets import Dataset


DATA_SOURCE = "DSBench_data_modeling"
DATA_PATH = "/data/download/dsbench/data_modeling"
INDEX_FILE = "/data/download/dsbench/data_modeling/data.json" # 74 challenges

TASK_PROMPT = """Save the final results as 'submission.csv'.

*** CRITICAL INSTRUCTIONS ***
1. MANDATORY OUTPUT: You MUST generate the 'submission.csv' file in the correct format. The task is incomplete without this file.
2. PURSUE EXTREME PERFORMANCE: Your goal is to achieve the highest possible leaderboard score."""

SAVE_PATH = "/data/hf-datasets/dsbench"


def generate_sample(sample_idx, 
                    instruction, 
                    expected_file, 
                    ground_truth_file, 
                    baseline_result,
                    gold_result,
                    workspace, 
                    evaluation_script):
    prompt = [
        {
            "role": "user",
            "content": instruction
        }
    ]

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": DATA_SOURCE,
        "agent_name": "deep_analyze_agent_loop",
        "reward_model": {
            "style": "rule",
            "expected_file": expected_file,
            "ground_truth": ground_truth_file,
            "baseline_result": baseline_result,
            "gold_result": gold_result
        },
        "extra_info": {
            "index": sample_idx,
            "evaluation_script": evaluation_script,
            "instruction": instruction,
            "workspace": workspace,
        },
    }

def main():
    sample_idx = 0
    samples = []

    tasks = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    for id in tqdm(range(len(tasks))):
        task = tasks[id]
        name = task["name"]

        description_file = os.path.join(DATA_PATH, "data", "task", f"{name}.txt")
        with open(description_file, "r", encoding="utf-8") as f:
            description = f.read()

        instruction = f"{description}\n\n{TASK_PROMPT}"
        workspace = os.path.join(DATA_PATH, "data", "data_resplit", name)
        expected_file = os.path.join(DATA_PATH, "data", "data_resplit", name, "submission.csv")
        ground_truth_file = os.path.join(DATA_PATH, "data", "answers", name, "test_answer.csv")
        evaluate_script = os.path.join(DATA_PATH, "evaluation", f"{name}_eval.py")
        baseline_file = os.path.join(DATA_PATH, "save_performance", "baseline", name, "result.txt")
        gold_file = os.path.join(DATA_PATH, "save_performance", "GT", name, "result.txt")

        with open(baseline_file, "r") as f:
            content = f.read().strip()
            baseline_result = float(content)

        with open(gold_file, "r") as f:
            content = f.read().strip()
            gold_result = float(content)
        
        sample = generate_sample(sample_idx=sample_idx, 
                                 instruction=instruction,
                                 expected_file=expected_file,
                                 ground_truth_file=ground_truth_file,
                                 baseline_result=baseline_result,
                                 gold_result=gold_result,
                                 workspace=workspace,
                                 evaluation_script=evaluate_script)
        samples.append(sample)
        sample_idx += 1
            
    final_dataset = Dataset.from_list(samples)

    os.makedirs(SAVE_PATH, exist_ok=True)
    final_dataset.to_parquet(os.path.join(SAVE_PATH, "data_modeling.parquet"))


if __name__ == "__main__":
    main()
