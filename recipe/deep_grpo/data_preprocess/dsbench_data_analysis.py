import os

from tqdm import tqdm
import json
import pandas as pd
from datasets import Dataset

import tiktoken


DATA_SOURCE = "DSBench_data_analysis"
DATA_PATH = "/data/download/dsbench/data_analysis/data"
INDEX_FILE = "/data/download/dsbench/data_analysis/data.json" # 38 challenges, each challenge has many questions, total question number is 466
SAVE_PATH = "/data/hf-datasets/dsbench"

TOKENS4GENERATION = 4000
MODEL_CONTEXT_LIMIT = 32768
ENCODING = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09")


def read_txt(path):
    with open(path, "r") as f:
        return f.read()

def find_excel_files(directory):
    excel_files = [
        file
        for file in os.listdir(directory)
        if (
            file.lower().endswith("xls")
            or file.lower().endswith("xlsx")
            or file.lower().endswith("xlsb")
            or file.lower().endswith("xlsm")
        )
        and "answer" not in file.lower()
    ]
    return excel_files if excel_files else None

def read_excel(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".xlsb":
        engine = "pyxlsb"
    elif ext in (".xlsx", ".xlsm"):
        engine = "openpyxl"
    elif ext == ".xls":
        engine = "xlrd"
    else:
        raise ValueError(f"Unsupported Excel extension: {ext} for file {file_path}")

    xls = pd.ExcelFile(file_path, engine=engine)

    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def combine_sheets(sheets):
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = df.to_string(index=False)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def normalize_answer(answer):
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)

def generate_sample(sample_idx, text, question, answer, workspace):
    instruction = text + f"The questions are detailed as follows.\n{question}"
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
        "reward_model": {"style": "rule", "ground_truth": normalize_answer(answer)},
        "agent_name": "deep_analyze_agent_loop",
        "extra_info": {
            "index": sample_idx,
            "question": question,
            "instruction": instruction,
            "workspace": workspace,
        },
    }


def main():


    sample_idx = 0
    samples = []

    challenges = []
    with open(INDEX_FILE, "r") as f:
        for line in f:
            challenges.append(eval(line.strip()))

    for id in tqdm(range(len(challenges))):
        challenge = challenges[id]
        if len(challenge["questions"]) > 0:
            
            # workspace
            workspace = os.path.join(DATA_PATH, challenge["id"])
            
            # introduction
            introduction = read_txt(
                os.path.join(workspace, "introduction.txt")
            )

            # questions
            questions = []
            for question_name in challenge["questions"]:
                questions.append(
                    read_txt(
                        os.path.join(workspace, question_name + ".txt")
                    )
                )

            # answers
            answers = challenge["answers"]
            
            # excel content
            excel_content = ""
            excel_files = find_excel_files(workspace)
            if excel_files:
                excel_blocks = []
                for excel_file in excel_files:
                    excel_file_path = os.path.join(workspace, excel_file)
                    try:
                        sheets = read_excel(excel_file_path) # Dict[sheet_name => dataframe]
                        combined_text = combine_sheets(sheets)
                        excel_blocks.append(f"The excel file {excel_file} is:\n{combined_text}")
                    except Exception as e:
                        print(f"Error reading file: {excel_file_path}")
                        print(f"Error details: {e}")
                        continue
                excel_content = "\n\n".join(excel_blocks)
                excel_content = ENCODING.decode(
                    ENCODING.encode(excel_content)[
                        TOKENS4GENERATION - MODEL_CONTEXT_LIMIT :
                    ]
                )
            
            text = ""
            if excel_content:
                text += f"The workbook is detailed as follows.\n{excel_content}\n"
            text += f"The introduction is detailed as follows.\n{introduction}\n"

            assert len(questions) == len(answers), f"Questions/answers length mismatch in challenge {challenge['id']}"
            for question, answer in zip(questions, answers):
                sample = generate_sample(sample_idx, text, question, answer, workspace)
                samples.append(sample)
                sample_idx += 1
            
    final_dataset = Dataset.from_list(samples)

    os.makedirs(SAVE_PATH, exist_ok=True)
    final_dataset.to_parquet(os.path.join(SAVE_PATH, "data_analysis.parquet"))

if __name__ == "__main__":
    main()
