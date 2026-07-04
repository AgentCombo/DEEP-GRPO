import os
import json
import argparse
from typing import Any, Dict, List

from datasets import Dataset
from recipe.deep_grpo.data_preprocess.tableqa.prompt import COT_PROMPT_MULTHIERTT_TEMPLATE


def format_tables_for_prompt(tables_data):
    """Format table data as a string representation for prompts"""
    result = []

    table_id = 0
    for table_name, table in tables_data.items():
        result.append(f"Table {table_id}:")
        table_id += 1

        formatted_table = []
        for row in table:
            formatted_table.append(" | ".join([str(cell) for cell in row]))

        result.append("\n".join(formatted_table))
        result.append("")

    return "\n".join(result)

def format_table_description(table_desc):
    """Format table description as a string"""
    result = []

    # 对每个表格描述项进行格式化
    for key, desc in table_desc.items():
        result.append(f"{desc}")

    return "\n".join(result)

def format_paragraphs(paragraphs):
    """Format paragraph data as a string"""
    if not paragraphs:
        return "No additional text provided."

    return "\n\n".join(
        [f"Paragraph {i+1}: {para}" for i, para in enumerate(paragraphs)]
    )

def create_prompt_from_multhiertt(item):
    """Create prompt for multihiertt item"""
    question = item.get("question", "")

    tables_str = format_tables_for_prompt(item.get("tables", {}))

    table_desc_str = format_table_description(item.get("table_description", {}))

    text_str = format_paragraphs(item.get("paragraphs", []))

    prompt = COT_PROMPT_MULTHIERTT_TEMPLATE.format(
        question=question,
        table=tables_str,
        table_description=table_desc_str,
        text=text_str,
    )

    return prompt

def process_fn(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = create_prompt_from_multhiertt(example)

    uid = example["uid"]
    assert type(uid) == str and len(uid) > 0

    question = example["question"]
    assert type(question) == str and len(question) > 0

    answer = example["answer"]
    assert answer is not None
    answer = str(answer)
    assert len(answer) > 0

    question_type = example["question_type"]
    assert type(question_type) == str and len(question_type) > 0

    data = {
        "data_source": "multihiertt",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "tableqa",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "agent_name": "reasoning_agent_loop",
        "extra_info": {
            "uid": uid,
            "question": question,
            "question_type": question_type,
        },
    }
    return data

def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiHierTT for Verl")
    parser.add_argument("--input_file", 
                        type=str, 
                        default="/data/download/tableqa/data/multihiertt/processed_test-dev_out_3_8.json",
                        help="Path to json file (JSON list)"
                      )
    parser.add_argument("--output_path", 
                        type=str, 
                        default="/data/hf-datasets/multihiertt",
                        help="Directory to save parquet")
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    print(f"Loaded {len(raw_items)} items from {args.input_file}.")

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
