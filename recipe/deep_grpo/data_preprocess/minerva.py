import os

import datasets

SAVE_PATH = os.environ.get("MINERVA_SAVE_PATH", "/data/hf-datasets/minerva")
TEST_DATASET_PATH = os.environ.get("MINERVA_TEST_DATASET_PATH", "/data/evaluation_suite/minerva")


test_dataset = datasets.load_from_disk(TEST_DATASET_PATH)
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."


def make_test_map_fn(split):
    def process_fn(example, idx):
        question = example["problem"]

        question = question + " " + instruction_following

        answer = example["answer"]

        if isinstance(answer, float) or isinstance(answer, int):
          answer = str(answer)

        if isinstance(answer, str):
            answer = [answer]

        data = {
            "data_source": "Minerva",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "agent_name": "reasoning_agent_loop",
            "extra_info": {"split": split, "index": idx, "question": question, "answer": answer},
        }
        return data

    return process_fn

test_dataset = test_dataset.map(function=make_test_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

os.makedirs(SAVE_PATH, exist_ok=True)
test_dataset.to_parquet(os.path.join(SAVE_PATH, "test.parquet"))
