"""Regression tests for teacher suffix EOS labels.

Run directly:
    PYTHONPATH=. python3 recipe/deep_grpo/tests/test_teacher_suffix_eos.py
"""

import asyncio
from types import SimpleNamespace
from unittest import SkipTest

from recipe.deep_grpo.teacher_suffix_utils import append_eos_if_missing


class _AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


class _DummyTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "shared suffix":
            return [10, 20, 30]
        raise AssertionError(f"unexpected text to encode: {text!r}")

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(tid) for tid in token_ids)


def _config():
    deep_grpo = _AttrDict(
        {
            "prefix_inject_mode": {"enabled": False},
            "teacher_suffix_synthesis": {"min_suffix_len": 0},
        }
    )
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(deep_grpo=deep_grpo)
        )
    )


async def _synthesize_with_fake_teacher():
    try:
        from recipe.deep_grpo.agent_loop import deep_grpo_agent_loop
        from recipe.deep_grpo.agent_loop.deep_grpo_agent_loop import DeepGRPOAgentLoop
        from recipe.deep_grpo.protocol import FailedTrajectoryEntry
        from recipe.deep_grpo.protocol import RewardInfo
    except ModuleNotFoundError as exc:
        raise SkipTest(
            f"TSAgentLoop dependencies are not installed locally: {exc.name}"
        ) from exc

    agent = DeepGRPOAgentLoop.__new__(DeepGRPOAgentLoop)
    agent.loop = asyncio.get_running_loop()
    agent.tokenizer = _DummyTokenizer()
    agent.config = _config()
    agent.max_model_len = 128
    agent.low_quality_trajectory_reward_threshold = 0.0

    async def _score_node(node):
        node.reward = 1.0
        node.reward_info = RewardInfo(reward=1.0, completed=1, finished=1)

    agent._score_node = _score_node

    async def _fake_call_teacher_with_retry(**_kwargs):
        return "shared suffix", "raw"

    original_call = deep_grpo_agent_loop.call_teacher_with_retry
    deep_grpo_agent_loop.call_teacher_with_retry = _fake_call_teacher_with_retry
    try:
        entry = FailedTrajectoryEntry(
            prompt_ids=[1, 2],
            response_ids=[10, 20],
            response_mask=[1, 1],
            data_instance={"data_source": "GSM8K"},
            tree_id="tree",
            num_turns=1.0,
        )
        return await agent.synthesize_teacher_suffix(
            entry,
            min_prefix_match_tokens=1,
            min_prefix_match_ratio=0.0,
        )
    finally:
        deep_grpo_agent_loop.call_teacher_with_retry = original_call


def test_append_eos_if_missing_adds_label_and_mask():
    ids, mask, appended = append_eos_if_missing(
        [30],
        [1],
        _DummyTokenizer(),
    )
    assert ids == [30, 99]
    assert mask == [1, 1]
    assert appended is True


def test_append_eos_if_missing_does_not_duplicate():
    ids, mask, appended = append_eos_if_missing(
        [30, 99],
        [1, 1],
        _DummyTokenizer(),
    )
    assert ids == [30, 99]
    assert mask == [1, 1]
    assert appended is False


def test_synthesize_teacher_suffix_appends_eos_label():
    result = asyncio.run(_synthesize_with_fake_teacher())
    assert result is not None
    assert result.teacher_suffix is not None
    assert result.teacher_suffix.suffix_ids == [30, 99]
    assert result.teacher_suffix.suffix_mask == [1, 1]


if __name__ == "__main__":
    test_append_eos_if_missing_adds_label_and_mask()
    test_append_eos_if_missing_does_not_duplicate()
    try:
        test_synthesize_teacher_suffix_appends_eos_label()
    except SkipTest as exc:
        print(f"SKIP test_synthesize_teacher_suffix_appends_eos_label: {exc}")
    print("test_teacher_suffix_eos PASSED")
