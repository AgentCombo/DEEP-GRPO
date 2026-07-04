import os
import json
import argparse
from typing import Dict, Any
from datasets import Dataset


def format_table_for_prompt(table_data):
    """Format table data into a string representation for prompts"""
    result = []

    # Add introduction information
    if "intro" in table_data and table_data["intro"]:
        result.append(f"Introduction: {table_data['intro']}")

    # Add section title and text
    if "section_title" in table_data and table_data["section_title"]:
        section_text = table_data.get("section_text", "")
        result.append(f"Section Description: {section_text}")

    # Add table content
    if "header" in table_data and table_data["header"] and "data" in table_data:
        # Get the header
        if table_data["header"][0]:
            headers = table_data["header"][0]
            # Format header
            header_row = " | ".join(headers)
            result.append(header_row)

        # Format data rows
        for row in table_data["data"]:
            result.append(" | ".join(row))

    return "\n".join(result)

def format_text_for_prompt(link_data):
    """Format linked text data into a string representation for prompts"""
    result = []

    for link, content in link_data.items():
        # Extract the last part of the link as the prefix
        link_parts = link.split("/")
        prefix = link_parts[-1] if link_parts else link

        # Add text with prefix
        result.append(f"{prefix}: {content}")

    return "\n\n".join(result)

def create_prompt_from_hybridqa(item, is_think):
    """Create prompt for a HybridQA item"""
    # Format table
    table_str = format_table_for_prompt(item["table"])

    # Format text
    text_str = format_text_for_prompt(item["relevant_links"])

    if is_think:
        from prompt_think import COT_PROMPT_HYBRIDQA_TEMPLATE

        # Generate prompt
        prompt = COT_PROMPT_HYBRIDQA_TEMPLATE.format(
            table=table_str, text=text_str, question=item["question"]
        )
    else:
        from prompt import COT_PROMPT_HYBRIDQA_TEMPLATE

        # Generate prompt
        prompt = COT_PROMPT_HYBRIDQA_TEMPLATE.format(
            table=table_str, text=text_str, question=item["question"]
        )

    return prompt

def process_fn(example: Dict[str, Any], args) -> Dict[str, Any]:
    prompt_text = create_prompt_from_hybridqa(example, is_think=args.think)

    qid = example["question_id"]
    assert type(qid) == str and len(qid) > 0

    question = example["question"]
    assert type(question) == str and len(question) > 0

    answer = example["answer"]
    assert type(answer) == str and len(answer) > 0

    data = {
        "data_source": "hybridqa",
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
            "question_id": qid,
            "instruction": question,
        },
    }

    return data


def main():
    parser = argparse.ArgumentParser(description="Preprocess HybridQA for Verl")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/download/tableqa/data/hybridqa/test_top_1_with_answers.json",
        help="Path to HybridQA json file (a JSON list)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/hf-datasets/hybridqa",
        help="Directory to save parquet files",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Use thinking prompt template",
    )

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # load
    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    print(f"Loaded {len(raw_items)} items from {args.input_file}.")

    processed = []
    for item in raw_items:
        out = process_fn(item, args)
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
