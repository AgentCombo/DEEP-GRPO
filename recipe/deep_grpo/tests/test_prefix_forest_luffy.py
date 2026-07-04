"""Focused tests for prefix-forest LUFFY teacher continuations.

Run directly:
    PYTHONPATH=. python3 recipe/deep_grpo/tests/test_prefix_forest_luffy.py
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class FakeTokenizer:
    eos_token_id = None

    def __init__(self):
        self.padding_side = "right"

    def pad(
        self,
        encoded_inputs,
        padding,
        max_length,
        return_tensors,
        return_attention_mask=True,
    ):
        assert padding == "max_length"
        assert return_tensors == "pt"
        rows = []
        masks = []
        for item in encoded_inputs:
            ids = list(item["input_ids"])
            pad_len = max_length - len(ids)
            assert pad_len >= 0
            if self.padding_side == "left":
                padded = [0] * pad_len + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + [0] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            rows.append(padded)
            masks.append(mask)
        out = {"input_ids": torch.tensor(rows, dtype=torch.long)}
        if return_attention_mask:
            out["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return out


@dataclass
class FakeRewardInfo:
    reward: float
    completed: int = 1
    finished: int = 1
    judgement_reply: str = ""


@dataclass
class FakeNode:
    augmented_prompt_ids: list
    teacher_suffix_ids: list
    teacher_suffix_mask: list
    teacher_suffix_reward: float
    teacher_suffix_reward_info: FakeRewardInfo
    num_turns: float = 1.0


def _trainer(node, teacher_loss_type="luffy", snis_beta=1.0):
    trainer = object.__new__(RayPPOTrainer)
    trainer.prefix_forest_luffy_enabled = True
    trainer.prefix_forest_teacher_loss_type = teacher_loss_type
    trainer.prefix_forest_snis_beta = snis_beta
    trainer._forest_luffy_tree_id_to_node = {
        "tid": ((1, 2), "node-a", node),
    }
    trainer.tokenizer = FakeTokenizer()
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                max_model_len=8,
                temperature=1.0,
                deep_grpo={"low_quality_trajectory_reward_threshold": 0.0},
            ),
        ),
    )
    return trainer


def _main_batch(rewards):
    n = len(rewards)
    response_mask = torch.tensor([[1, 1, 0, 0]] * n, dtype=torch.long)
    token_scores = torch.zeros((n, 4), dtype=torch.float32)
    for i, reward in enumerate(rewards):
        token_scores[i, 1] = float(reward)
    advantages = torch.full((n, 4), 99.0, dtype=torch.float32) * response_mask
    batch = TensorDict(
        {
            "prompts": torch.ones((n, 4), dtype=torch.long),
            "responses": torch.ones((n, 4), dtype=torch.long),
            "response_mask": response_mask,
            "input_ids": torch.ones((n, 8), dtype=torch.long),
            "attention_mask": torch.ones((n, 8), dtype=torch.long),
            "position_ids": torch.arange(8).repeat(n, 1),
            "token_level_scores": token_scores,
            "token_level_rewards": token_scores.clone(),
            "advantages": advantages.clone(),
            "returns": advantages.clone(),
        },
        batch_size=n,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "__tree_ids__": np.array(["tid"] * n, dtype=object),
            "__node_ids__": np.array([f"student-{i}" for i in range(n)], dtype=object),
            "__reward_infos__": np.array(
                [FakeRewardInfo(reward=float(r)) for r in rewards],
                dtype=object,
            ),
        },
        meta_info={"metrics": [{} for _ in rewards]},
    )


def test_prefix_luffy_reweights_students_and_builds_teacher_row():
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node)
    main_batch = _main_batch([0.0, 0.0, 0.0])
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 1
    assert metrics["prefix_luffy/teacher_rows_built"] == 1
    assert metrics["prefix_luffy/student_rows_reweighted"] == 3
    assert np.isclose(metrics["prefix_luffy/teacher_advantage_mean"], 0.75)
    # Leave-one-out: all-fail students use the STUDENT-ONLY baseline
    # (mean[0,0,0]=0), so their advantage is exactly 0 (will be filtered
    # downstream), NOT the mixed-baseline -0.25. The teacher row still uses the
    # mixed baseline (teacher_adv = 1 - 0.25 = 0.75) and is injected.
    assert np.allclose(
        main_batch.batch["advantages"][:, :2].numpy(),
        np.zeros((3, 2)),
    )
    assert np.allclose(
        teacher_batch.batch["advantages"][0, :2].numpy(),
        np.full(2, 0.75),
    )
    assert teacher_batch.non_tensor_batch["__tree_ids__"][0] == "tid"


def test_prefix_luffy_all_equal_skips_zero_adv_teacher_row():
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node)
    main_batch = _main_batch([1.0, 1.0, 1.0])
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/equal_reward_groups_skipped"] == 1
    assert torch.count_nonzero(main_batch.batch["advantages"]).item() == 0


def test_prefix_luffy_missing_suffix_keeps_student_rollouts_only():
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[],
        teacher_suffix_mask=[],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node)
    main_batch = _main_batch([0.0, 0.0, 0.0])
    before = main_batch.batch["advantages"].clone()
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/missing_suffix_skipped"] == 1
    assert torch.equal(main_batch.batch["advantages"], before)


def test_prefix_luffy_missing_reward_keeps_student_rollouts_only():
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=None,
        teacher_suffix_reward_info=None,
    )
    trainer = _trainer(node)
    main_batch = _main_batch([0.0, 0.0, 0.0])
    before = main_batch.batch["advantages"].clone()
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/missing_reward_skipped"] == 1
    assert torch.equal(main_batch.batch["advantages"], before)


def test_low_reward_teacher_gated_before_baseline_rewrite():
    """Declaration ④: a wrong teacher (R*=0) must not touch student advantages,
    even when students disagree among themselves (mixed group)."""
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=0.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=0.0),
    )
    trainer = _trainer(node, teacher_loss_type="snis")
    main_batch = _main_batch([1.0, 0.0, 0.0])  # mixed: not equal-reward filtered
    before = main_batch.batch["advantages"].clone()
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/low_reward_teacher_skipped"] == 1
    assert metrics["prefix_luffy/student_rows_reweighted"] == 0
    assert torch.equal(main_batch.batch["advantages"], before)


def test_snis_weight_value_and_bound():
    """w̃ = exp(A*/β)/mean_j exp(A_j/β); all-fail group n=3, teacher correct:
    baseline=0.25, w̃ = e^0.75 / ((3·e^-0.25 + e^0.75)/4) ≈ 1.9015 ≤ e."""
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node, teacher_loss_type="snis", snis_beta=1.0)
    main_batch = _main_batch([0.0, 0.0, 0.0])
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    b = 0.25
    exp_advs = [np.exp(0.0 - b)] * 3 + [np.exp(1.0 - b)]
    expected_w = np.exp(1.0 - b) / np.mean(exp_advs)
    assert len(teacher_batch) == 1
    assert np.isclose(metrics["prefix_luffy/snis_weight_mean"], expected_w)
    assert metrics["prefix_luffy/snis_weight_max"] <= np.e + 1e-9
    # weight rides the advantages tensor (masked to valid tokens)
    assert np.allclose(
        teacher_batch.batch["advantages"][0, :2].numpy(),
        np.full(2, expected_w),
    )
    # students use the student-only (leave-one-out) baseline; all-fail group
    # -> advantage exactly 0 (filtered downstream). The teacher w̃ still uses
    # the mixed baseline (b=0.25), unchanged.
    assert np.allclose(
        main_batch.batch["advantages"][:, :2].numpy(),
        np.zeros((3, 2)),
    )


def test_snis_baseline_shift_invariance():
    """Same reward GAP, shifted rewards → identical w̃ (softmax property)."""
    def run(student_r, teacher_r):
        node = FakeNode(
            augmented_prompt_ids=[1, 2, 3],
            teacher_suffix_ids=[9, 10],
            teacher_suffix_mask=[1, 1],
            teacher_suffix_reward=teacher_r,
            teacher_suffix_reward_info=FakeRewardInfo(reward=teacher_r),
        )
        trainer = _trainer(node, teacher_loss_type="snis")
        metrics = {}
        trainer._attach_prefix_forest_luffy_teacher_continuations(
            _main_batch([student_r] * 3),
            metrics,
        )
        return metrics["prefix_luffy/snis_weight_mean"]

    assert np.isclose(run(0.0, 1.0), run(2.0, 3.0))


def test_teacher_retires_when_any_student_succeeds():
    """k>=1 retirement (theory: damage regime requires k>=1; teacher must exit).
    Mixed group [1,0,0] + teacher 1.0: students keep their student-only
    (leave-one-out) advantages, but NO teacher row is built."""
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node, teacher_loss_type="snis")
    main_batch = _main_batch([1.0, 0.0, 0.0])
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/teacher_not_better_skipped"] == 1
    assert metrics["prefix_luffy/student_rows_reweighted"] == 3
    # student-only baseline = mean(1,0,0) = 1/3, NOT the mixed 0.5
    assert np.allclose(
        main_batch.batch["advantages"][0, :2].numpy(), np.full(2, 2.0 / 3.0)
    )
    assert np.allclose(
        main_batch.batch["advantages"][1, :2].numpy(), np.full(2, -1.0 / 3.0)
    )


def test_none_arm_rewrites_baseline_but_builds_no_teacher_rows():
    """Ablation arm (teacher_loss_type=None): no teacher row is trained. Under
    leave-one-out, all-fail injected groups get student advantage 0 (filtered),
    so the none arm is pure state-curriculum GRPO."""
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node, teacher_loss_type=None)
    main_batch = _main_batch([0.0, 0.0, 0.0])
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch,
        metrics,
    )

    assert len(teacher_batch) == 0
    assert metrics["prefix_luffy/student_rows_reweighted"] == 3
    # Leave-one-out student baseline: all-fail group -> advantage exactly 0
    # (filtered downstream). With no teacher row either, the none arm is pure
    # state-curriculum GRPO: all-fail injected nodes contribute no gradient,
    # exactly as teacher-free GRPO would handle an all-fail group.
    assert np.allclose(
        main_batch.batch["advantages"][:, :2].numpy(),
        np.zeros((3, 2)),
    )


def test_allfail_students_get_zero_adv_but_teacher_injected():
    """Regression for the leave-one-out fix: in an all-fail group the students
    must get advantage EXACTLY 0 (so the downstream zero-advantage filter drops
    them — they are junk rollouts that should not enter the gradient), while the
    teacher row IS still injected (the all-fail node is exactly where the
    teacher signal matters). Before the fix, the mixed baseline gave all-fail
    students a small nonzero advantage that slipped past the filter."""
    node = FakeNode(
        augmented_prompt_ids=[1, 2, 3],
        teacher_suffix_ids=[9, 10],
        teacher_suffix_mask=[1, 1],
        teacher_suffix_reward=1.0,
        teacher_suffix_reward_info=FakeRewardInfo(reward=1.0),
    )
    trainer = _trainer(node, teacher_loss_type="snis")
    main_batch = _main_batch([0.0, 0.0, 0.0])  # all-fail
    metrics = {}

    teacher_batch = trainer._attach_prefix_forest_luffy_teacher_continuations(
        main_batch, metrics
    )

    # students: advantage exactly 0 -> would be filtered by the >1e-8 gate
    adv = main_batch.batch["advantages"].numpy()
    assert np.allclose(adv, 0.0), "all-fail students must have zero advantage"
    # teacher row still built (desert keeps its signal)
    assert len(teacher_batch) == 1
    assert metrics["prefix_luffy/teacher_rows_built"] == 1


if __name__ == "__main__":
    test_prefix_luffy_reweights_students_and_builds_teacher_row()
    print("test_prefix_luffy_reweights_students_and_builds_teacher_row PASSED")
    test_prefix_luffy_all_equal_skips_zero_adv_teacher_row()
    print("test_prefix_luffy_all_equal_skips_zero_adv_teacher_row PASSED")
    test_prefix_luffy_missing_suffix_keeps_student_rollouts_only()
    print("test_prefix_luffy_missing_suffix_keeps_student_rollouts_only PASSED")
    test_prefix_luffy_missing_reward_keeps_student_rollouts_only()
    print("test_prefix_luffy_missing_reward_keeps_student_rollouts_only PASSED")
    test_low_reward_teacher_gated_before_baseline_rewrite()
    print("test_low_reward_teacher_gated_before_baseline_rewrite PASSED")
    test_snis_weight_value_and_bound()
    print("test_snis_weight_value_and_bound PASSED")
    test_snis_baseline_shift_invariance()
    print("test_snis_baseline_shift_invariance PASSED")
    test_teacher_retires_when_any_student_succeeds()
    print("test_teacher_retires_when_any_student_succeeds PASSED")
    test_allfail_students_get_zero_adv_but_teacher_injected()
    print("test_allfail_students_get_zero_adv_but_teacher_injected PASSED")
    test_none_arm_rewrites_baseline_but_builds_no_teacher_rows()
    print("test_none_arm_rewrites_baseline_but_builds_no_teacher_rows PASSED")
