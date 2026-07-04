import os

import argparse
from pathlib import Path
from typing import Dict, Any, List

from huggingface_hub import hf_hub_download
from datasets import load_dataset


SOLVE_PROMPT_TEMPLATE = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


def prepare_context_files(data_dir: str):
  CONTEXT_FILENAMES = [
    "data/context/acquirer_countries.csv",
    "data/context/payments-readme.md",
    "data/context/payments.csv",
    "data/context/merchant_category_codes.csv",
    "data/context/fees.json",
    "data/context/merchant_data.json",
    "data/context/manual.md",
  ]
  for filename in CONTEXT_FILENAMES:
      hf_hub_download(
          repo_id="adyen/DABstep",
          repo_type="dataset",
          filename=filename,
          local_dir=data_dir,
          force_download=True,
          endpoint=os.environ.get("HF_ENDPOINT"),
      )

def collect_context_paths(context_root_dir: str) -> List[str]:
    filenames = [
        "data/context/acquirer_countries.csv",
        "data/context/payments-readme.md",
        "data/context/payments.csv",
        "data/context/merchant_category_codes.csv",
        "data/context/fees.json",
        "data/context/merchant_data.json",
        "data/context/manual.md",
    ]
    paths = []
    for rel in filenames:
        p = (Path(context_root_dir) / rel).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing context file: {p}")
        paths.append(p.as_posix())
    return paths


def build_prompt(question: str, guidelines: str, context_paths: List[str]) -> str:
    context_files_str = "\n".join(context_paths)
    return SOLVE_PROMPT_TEMPLATE.format(
        context_files=context_files_str,
        question=question,
        guidelines=guidelines,
    )


def process_fn(
    example: Dict[str, Any],
    context_paths: List[str],
) -> Dict[str, Any]:
    assert "task_id" in example
    assert example["question"]
    assert example["guidelines"]

    task_id = str(example["task_id"])
    question = str(example["question"])
    guidelines = str(example["guidelines"])

    prompt = build_prompt(question, guidelines, context_paths)

    return {
        "data_source": "dabstep",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "data_analysis",
        "reward_model": {
            "style": "rule",
            "ground_truth": "", # need to submit to the leaderboard to get the score
        },
        "agent_name": "deep_analyze_agent_loop",
        "extra_info": {
            "instruction": question,
            "task_id": task_id,
            "workspace": f"/tmp/dabstep/{task_id}",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_dir", type=str, default="/data/download/DABstep")
    parser.add_argument("--save_path", type=str, default="/data/hf-datasets/dabstep")
    parser.add_argument("--download_context", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.download_context:
        os.makedirs(args.context_dir, exist_ok=True)
        prepare_context_files(args.context_dir)

    context_paths = collect_context_paths(args.context_dir)

    hf_dataset = load_dataset("adyen/DABstep", name="tasks", split="default")
    print(f"Loaded {len(hf_dataset)} tasks.")

    def _map_fn(example):
        return process_fn(example, context_paths=context_paths)

    processed_dataset = hf_dataset.map(
        _map_fn,
        remove_columns=hf_dataset.column_names,
        desc="Processing DABstep data",
    )

    processed_dataset.to_parquet(os.path.join(args.save_path, "test.parquet"))


if __name__ == "__main__":
    main()
