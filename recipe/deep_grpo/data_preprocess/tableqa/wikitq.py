import os
import argparse
from datasets import Dataset
import json


def create_prompt_from_wikitq(
    item, is_think=False, table_format="markdown", perturbation="none"
):
    """
    Create prompt for WikiTableQuestion item with different table formats and perturbations

    Args:
        item: The input item containing table and question
        is_think: Whether to use thinking prompt template
        table_format: Format for table representation - "markdown", "csv", or "dataframe"
        perturbation: Type of table perturbation - "none", "row_shuffle", "col_shuffle", "both_shuffle"

    Returns:
        Formatted prompt string
    """
    table_str = ""
    if "table" in item:
        # Deep copy table data to avoid modifying original data
        import copy
        import random

        table = copy.deepcopy(item["table"])

        # Get headers and data rows
        header = table[0] if len(table) > 0 else []
        data_rows = table[1:] if len(table) > 1 else []

        # Apply perturbation
        row_shuffle = perturbation in ["row_shuffle", "both_shuffle"]
        col_shuffle = perturbation in ["col_shuffle", "both_shuffle"]

        # Apply column perturbation
        if col_shuffle and len(header) > 1:
            # Generate new column order
            col_indices = list(range(len(header)))
            random.shuffle(col_indices)

            # Reorder headers and data rows
            header = [header[i] for i in col_indices]
            data_rows = [
                [row[i] if i < len(row) else "" for i in col_indices]
                for row in data_rows
            ]

        # Apply row perturbation
        if row_shuffle and len(data_rows) > 1:
            # Keep headers unchanged, randomly shuffle data rows
            random.shuffle(data_rows)

        # Recombine table
        shuffled_table = [header] + data_rows

        # Generate table string based on selected format
        if table_format == "markdown":
            # Create Markdown format table string representation
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        elif table_format == "csv":
            # Create CSV format table string representation
            for row in shuffled_table:
                table_str += ",".join([f'"{str(cell)}"' for cell in row]) + "\n"

        elif table_format == "dataframe":
            # Import pandas library
            import pandas as pd

            # Create DataFrame-like table representation
            if len(shuffled_table) > 0:
                # Get headers and data
                headers = shuffled_table[0]
                data = shuffled_table[1:] if len(shuffled_table) > 1 else []

                # Create pandas DataFrame
                df = pd.DataFrame(data, columns=headers)

                # Configure pandas display options to show complete table
                pd.set_option("display.expand_frame_repr", False)
                pd.set_option("display.width", 500000)
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.max_colwidth", None)

                # Convert DataFrame to string
                table_str = df.to_string(index=True)

                # Add DataFrame information line
                table_str += f"\n\n[{len(data)} rows x {len(headers)} columns]\n"

                pd.reset_option("display.expand_frame_repr")
                pd.reset_option("display.width")
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
                pd.reset_option("display.max_colwidth")

        else:
            # Default use Markdown format
            for row in shuffled_table:
                table_str += " | ".join([str(cell) for cell in row]) + "\n"

        # If perturbation was applied, add explanatory note
        if perturbation != "none" and not is_think:  # Add note in non-thinking mode
            notes = []
            if row_shuffle:
                notes.append("Row order has been randomly shuffled (except header)")
            if col_shuffle:
                notes.append("Column order has been randomly shuffled")

            if notes:
                table_str += "\nNote: " + ", ".join(notes) + "\n"

    # Use different templates based on thinking requirement
    if is_think:
        from recipe.deep_grpo.data_preprocess.tableqa.prompt_think import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])
    else:
        from recipe.deep_grpo.data_preprocess.tableqa.prompt import COT_PROMPT_TEMPLATE

        prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])

    return prompt

def process_fn(example, idx, args):
    prompt_text = create_prompt_from_wikitq(
        example, 
        is_think=args.think, 
        table_format=args.table_format, 
        perturbation=args.perturbation
    )

    ground_truth = example["answer"]
    assert type(ground_truth) == str and len(ground_truth) > 0

    question = example["question"]
    assert type(question) == str and len(question) > 0

    table = example["table"]
    assert type(table) == list

    data = {
        "data_source": "wikitq",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "ability": "table-qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "agent_name": "reasoning_agent_loop",
        "extra_info": {
            "index": idx,
            "question": question,
        },
    }
    return data

def main():
    parser = argparse.ArgumentParser(description="Preprocess WikiTableQuestions for Verl")
    parser.add_argument("--local_dir", type=str, default="/data/download/tableqa/data/wikitq", help="Local directory containing jsonl files")
    parser.add_argument("--output_path", type=str, default="/data/hf-datasets/wikitq", help="Path to save parquet files")
    parser.add_argument("--table_format", type=str, default="markdown", choices=["markdown", "csv", "dataframe"])
    parser.add_argument("--perturbation", type=str, default="none", choices=["none", "row_shuffle", "col_shuffle", "both_shuffle"])
    parser.add_argument("--think", action="store_true", help="Use thinking prompt template")
    
    args = parser.parse_args()

    data_files = {}
    if os.path.exists(os.path.join(args.local_dir, "train.jsonl")):
        data_files["train"] = os.path.join(args.local_dir, "train.jsonl")
    if os.path.exists(os.path.join(args.local_dir, "test.jsonl")):
        data_files["test"] = os.path.join(args.local_dir, "test.jsonl")

    assert len(data_files.items()) == 1

    for split, file_path in data_files.items():
        print(f"Processing {split} from {file_path}...")
        
        processed_data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                processed_item = process_fn(item, idx, args)
                gt = processed_item["reward_model"]["ground_truth"]
                assert gt is not None and len(str(gt).strip()) > 0
                processed_data_list.append(processed_item)

        print(f"Loaded and processed {len(processed_data_list)} examples for {split}.")

        hf_dataset = Dataset.from_list(processed_data_list)

        hf_dataset = hf_dataset.shuffle(seed=42)
        train_dataset = hf_dataset.select(range(0, 1000))
        test_dataset = hf_dataset.select(range(1000, 1100))

        train_dataset.to_parquet(os.path.join(args.output_path, f"train.parquet"))
        test_dataset.to_parquet(os.path.join(args.output_path, f"test.parquet"))

if __name__ == "__main__":
    main()
