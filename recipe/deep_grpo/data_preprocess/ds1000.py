import os
from datasets import load_dataset

DATA_SOURCE = "/data/download/DS-1000"
SAVE_PATH = "/data/hf-datasets/ds1000"

SYS_PROMPT = """
Write a short code following the given format and indentation. Only provide the code completion needed. Don't repeat the context code.
Your response should end with:
Answer:
```python
<your_code_here>
```"""

def process_fn(example, idx):
    raw_prompt = example["prompt"]
    metadata = example["metadata"]
    reference_code = example["reference_code"]
    code_context = example["code_context"]
    library = metadata["library"] # 例如 numpy, pandas
    problem_id = int(metadata["problem_id"]) 

    prompt = (
        raw_prompt
        .replace("BEGIN SOLUTION\n<code>", "")
        .replace("<code>", "```python")
        .replace("</code>", "```")
        + SYS_PROMPT
    )
    
    data = {
        "data_source": "ds1000",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "code", 
        "reward_model": {
            "style": "rule", 
            "ground_truth": reference_code
        },
        "agent_name": "reasoning_agent_loop",
        "extra_info": {
            "index": idx,
            "library": library,
            "problem_id": problem_id,
            "code_context": code_context # used for evaluate
        },
    }
    return data

def main():
    hf_dataset = load_dataset("/data/download/DS-1000")["test"]
    
    print(f"Total examples: {len(hf_dataset)}")

    processed_dataset = hf_dataset.map(
        process_fn,
        with_indices=True,
        remove_columns=hf_dataset.column_names,
        desc="Processing DS-1000 data"
    )

    processed_dataset = processed_dataset.filter(
        lambda x: x["reward_model"]["ground_truth"] is not None and len(x["reward_model"]["ground_truth"].strip()) > 0
    )

    processed_dataset.to_parquet(os.path.join(SAVE_PATH, "full.parquet"))

    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)    
    print(f"Split sizes -> Train: {len(split_dataset['train'])}, Test: {len(split_dataset['test'])}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    split_dataset["train"].to_parquet(os.path.join(SAVE_PATH, "train.parquet"))
    split_dataset["test"].to_parquet(os.path.join(SAVE_PATH, "test.parquet"))
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()