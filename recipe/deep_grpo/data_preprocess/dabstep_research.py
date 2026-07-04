import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download
from datasets import load_dataset


def collect_file_info(
    directory: str,
) -> str:
    allowed_suffixes = {".csv", ".xlsx", ".sqlite", ".md", ".json"}

    dir_path = Path(directory) / "data" / "context" # TODO: assume the actual context files are under data/context/ 
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"CONTEXT_DIR not found or not a directory: {directory}")

    files = sorted(
        [
            f
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]
    )

    all_file_info_str = ""
    for idx, file_path in enumerate(files, start=1):
        size_bytes = file_path.stat().st_size
        size_kb = size_bytes / 1024
        file_info = {"name": file_path.resolve().as_posix(), "size": f"{size_kb:.1f}KB"}
        file_info_str = json.dumps(file_info, indent=4, ensure_ascii=False)
        all_file_info_str += f"File {idx}:\n{file_info_str}\n\n"

    return all_file_info_str.strip()

def build_prompt(question: str, context_files: str) -> str:
    return f"# Instruction\n{question}\n\n# Data\n{context_files}"

def process_fn(example: Dict[str, Any], context_files: str) -> Dict[str, Any]:
    assert "id" in example
    assert example["question"]
    assert example["type"]

    task_id = str(example["id"])
    question = str(example["question"])
    _type = str(example["type"])

    prompt = build_prompt(example["question"], context_files)

    data = {
        "data_source": "dabstep_research",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "data_analysis",
        "reward_model": {
            "style": "rule",
            "ground_truth": example.get("checklist", ""),
        },
        "agent_name": "deep_analyze_agent_loop",
        "extra_info": {
            "instruction": question,
            "task_id": task_id,
            "type": _type,
            "workspace": f"/tmp/dabstep_research/{task_id}"
        },
    }
    return data


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/data/download/dabstep-research/dabstep_research.jsonl")
    parser.add_argument("--context_dir", type=str, default="/data/download/DABstep")
    parser.add_argument("--save_path", type=str, default="/data/hf-datasets/dabstep_research")
    parser.add_argument("--download_context", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.download_context:
        os.makedirs(args.context_dir, exist_ok=True)
        prepare_context_files(args.context_dir)

    context_files = collect_file_info(args.context_dir)

    hf_dataset = load_dataset("json", data_files=args.jsonl_path, split="train")
    print(f"Loaded {len(hf_dataset)} tasks from {args.jsonl_path}")

    def _map_fn(example):
        return process_fn(example, context_files=context_files)

    processed_dataset = hf_dataset.map(
        _map_fn,
        remove_columns=hf_dataset.column_names,
        desc="Processing DABstep research data",
    )

    processed_dataset.to_parquet(os.path.join(args.save_path, "test.parquet"))


if __name__ == "__main__":
    main()
