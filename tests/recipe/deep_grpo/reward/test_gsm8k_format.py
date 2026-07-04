from dataclasses import dataclass
import sys
import types


def _import_gsm8k():
    try:
        from recipe.deep_grpo.reward import gsm8k
        return gsm8k
    except ModuleNotFoundError as exc:
        if exc.name != "pandas":
            raise

    @dataclass
    class RewardInfo:
        reward: float
        completed: int
        finished: int = 0
        judgement_reply: str | None = None

    protocol = types.ModuleType("recipe.deep_grpo.protocol")
    protocol.RewardInfo = RewardInfo
    sys.modules["recipe.deep_grpo.protocol"] = protocol
    sys.modules.pop("recipe.deep_grpo.reward.gsm8k", None)
    from recipe.deep_grpo.reward import gsm8k
    return gsm8k


gsm8k = _import_gsm8k()


def test_prompt_hash_marker_does_not_count_as_repeated_answer():
    text = (
        "<|im_start|>user\nWhat is 2 + 3? "
        "Let's think step by step and output the final answer after \"####\"."
        "<|im_end|>\n<|im_start|>assistant\n"
        "2 + 3 = 5.\n\n#### 5"
    )
    result = gsm8k.compute_score(text, "5")
    assert result.reward == 1.0
    assert result.completed == 1


def test_single_hash_and_single_boxed_are_allowed():
    text = "2 + 3 = 5.\n\n#### 5\n\nThe final answer is \\boxed{5}."
    result = gsm8k.compute_score(text, "5")
    assert result.reward == 1.0
    assert result.completed == 1


def test_repeated_hash_answer_marker_is_rejected():
    text = "2 + 3 = 5.\n\n#### 5\n\n#### 5"
    result = gsm8k.compute_score(text, "5")
    assert result.reward == 0
    assert result.completed == 0


def test_repeated_boxed_marker_is_rejected():
    text = "2 + 3 = 5.\n\n\\boxed{5}\n\nThe final answer is \\boxed{5}."
    result = gsm8k.compute_score(text, "5")
    assert result.reward == 0
    assert result.completed == 0


if __name__ == "__main__":
    test_prompt_hash_marker_does_not_count_as_repeated_answer()
    test_single_hash_and_single_boxed_are_allowed()
    test_repeated_hash_answer_marker_is_rejected()
    test_repeated_boxed_marker_is_rejected()
    print("test_gsm8k_format PASSED")
