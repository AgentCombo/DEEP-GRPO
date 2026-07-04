import os
import json
import argparse
from typing import Any, Dict, List

from datasets import Dataset

from recipe.deep_grpo.data_preprocess.tableqa.prompt import COT_PROMPT_FINQA_TEMPLATE


def format_table_for_prompt(table):
    """Format a table into a string for the prompt"""
    formatted_table = ""
    for row in table:
        formatted_table += " | ".join(str(cell) for cell in row) + "\n"
    return formatted_table.strip()

def format_text_list_for_prompt(text_list):
    """Format a list of text items into a string for the prompt"""
    if not text_list:
        return ""
    return "\n".join(text_list)

def create_prompt_from_finqa(item):
    """Create prompt for FinQA item"""
    # Format table
    table_str = format_table_for_prompt(item.get("table", []))

    # Format pre_text and post_text
    pre_text_str = format_text_list_for_prompt(item.get("pre_text", []))
    post_text_str = format_text_list_for_prompt(item.get("post_text", []))

    # Create prompt
    prompt = COT_PROMPT_FINQA_TEMPLATE.format(
        pre_text=pre_text_str,
        table=table_str,
        post_text=post_text_str,
        question=item.get("qa", {}).get("question", ""),
    )

    return prompt

def process_fn(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = create_prompt_from_finqa(example)

    qid = example["id"]
    assert type(qid) == str and len(qid) > 0

    qa = example["qa"]

    question = qa["question"]
    assert type(question) == str and len(question) > 0

    answer = qa["answer"]
    assert answer is not None
    answer = str(answer)
    if len(answer) == 0:
        answer = str(qa["exe_ans"])
    assert len(answer) > 0

    data = {
        "data_source": "finqa",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "tableqa",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "agent_name": "reasoning_agent_loop",
        "extra_info": {
            "question_id": qid,
            "question": question,
        },
    }
    return data

def main():
    parser = argparse.ArgumentParser(description="Preprocess FinQA for Verl")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/download/tableqa/data/finqa/test.json",
        help="Path to FinQA json file (JSON list).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/hf-datasets/finqa",
        help="Directory to save parquet.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"Loaded {len(raw_items)} items from {args.input_file}")

    processed: List[Dict[str, Any]] = []
    for item in raw_items:
        out = process_fn(item)
        gt = out["reward_model"]["ground_truth"]
        assert gt is not None and len(str(gt).strip()) > 0
        processed.append(out)

    hf_dataset = Dataset.from_list(processed)
    
    hf_dataset = hf_dataset.shuffle(seed=42)
    train_dataset = hf_dataset.select(range(0, 1000))
    test_dataset = hf_dataset.select(range(1000, 1100))

    train_dataset.to_parquet(os.path.join(args.output_path, f"train.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_path, f"test.parquet"))


if __name__ == "__main__":
    main()
