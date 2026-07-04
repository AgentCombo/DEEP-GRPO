import os
import datasets


SAVE_PATH = "/data/hf-datasets/deepscaler"
TRAIN_DATASET_PATH = "agentica-org/DeepScaleR-Preview-Dataset"


train_dataset = datasets.load_dataset(TRAIN_DATASET_PATH)["train"]

instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

def make_train_map_fn(split):
    def process_fn(example, idx):
        problem = example["problem"]

        question = problem + " " + instruction_following

        answer = example["answer"]

        if isinstance(answer, float) or isinstance(answer, int):
          answer = str(answer)

        if isinstance(answer, str):
            answer = [answer]

        data = {
            "data_source": "DeepScaleR",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "agent_name": "reasoning_agent_loop",
            "extra_info": {"split": split, "index": idx, "question": problem, "answer": answer}
        }
        return data

    return process_fn


train_dataset = train_dataset.map(function=make_train_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)

os.makedirs(SAVE_PATH, exist_ok=True)
train_dataset.to_parquet(os.path.join(SAVE_PATH, "train.parquet"))
