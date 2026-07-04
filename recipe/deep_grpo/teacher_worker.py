"""Background teacher annotation worker.

Runs in a separate daemon thread with its own asyncio event loop.
Pulls failed trajectories from FailedTrajectoryPool, calls teacher model
to synthesize correct suffixes, and deposits annotated BranchPointEntry
objects into TeacherAnnotatedPool.

Zero impact on training speed -- completely decoupled.
"""

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from recipe.deep_grpo.pools.failed_trajectory_pool import FailedTrajectoryPool
from recipe.deep_grpo.pools.teacher_annotated_pool import TeacherAnnotatedPool
from recipe.deep_grpo.pools.synthetic_prompt_pool import SyntheticPromptPool
from recipe.deep_grpo.pools.prefix_chain_pool import PrefixChainPool
from recipe.deep_grpo.pools.prefix_forest_pool import FailedPrefixEvent, PrefixForestPool
from recipe.deep_grpo.protocol import (
    BranchPointEntry,
    FailedTrajectoryEntry,
    SyntheticPromptEntry,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherAnnotationWorker:
    """Background worker that annotates failed trajectories with teacher suffixes.

    Architecture:
    - Runs in a daemon thread (auto-cleaned on process exit)
    - Has its own asyncio event loop (no interference with training loop)
    - Pulls from FailedTrajectoryPool (shared with training loop, thread-safe)
    - Pushes to TeacherAnnotatedPool (shared with training loop, thread-safe)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer,
        failed_pool: Optional[FailedTrajectoryPool],
        annotated_pool: Optional[TeacherAnnotatedPool],
        agent_loop_class,
        agent_loop_config,
        server_manager=None,
        synthetic_pool: Optional[SyntheticPromptPool] = None,
        chain_pool: Optional[PrefixChainPool] = None,
        forest_pool: Optional[PrefixForestPool] = None,
    ):
        # Chain/forest modes are mutually exclusive with the other output sinks;
        # they also replace failed_pool / annotated_pool as the work source.
        if chain_pool is not None or forest_pool is not None:
            assert synthetic_pool is None, (
                "chain_pool/forest_pool is mutually exclusive with synthetic_pool"
            )
            assert annotated_pool is None, (
                "chain_pool/forest_pool is mutually exclusive with annotated_pool"
            )
        assert not (chain_pool is not None and forest_pool is not None), (
            "chain_pool and forest_pool are mutually exclusive"
        )

        self.config = config
        self.tokenizer = tokenizer
        self.failed_pool = failed_pool
        self.annotated_pool = annotated_pool
        self.synthetic_pool = synthetic_pool   # set in prefix_inject flat mode
        self.chain_pool = chain_pool           # set in prefix_inject chain mode
        self.forest_pool = forest_pool         # set in prefix_inject forest mode
        self.agent_loop_class = agent_loop_class
        self.agent_loop_config = agent_loop_config
        self.server_manager = server_manager

        self.max_concurrent = int(config.get("max_concurrent_annotations", 64))
        assert self.max_concurrent > 0, (
            "teacher_suffix_synthesis.max_concurrent_annotations must be > 0, "
            f"got {self.max_concurrent}"
        )
        self.poll_interval = float(config.get("poll_interval", 2.0))
        assert self.poll_interval >= 0.0, (
            "teacher_suffix_synthesis.poll_interval must be >= 0.0, "
            f"got {self.poll_interval}"
        )
        self.min_prefix_match_tokens = config.get("min_prefix_match_tokens", 10)
        self.min_prefix_match_ratio = config.get("min_prefix_match_ratio", 0.05)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Metrics (thread-safe via GIL for simple int increments)
        self.total_attempted = 0
        self.total_succeeded = 0
        # Per-step delta tracking (reset by stats property each read)
        self._last_read_attempted = 0
        self._last_read_succeeded = 0

        # Chain-mode specific: cache of drained deepening requests awaiting
        # dispatch (we drain in batches from the pool but fire one task per
        # semaphore slot).
        self._chain_pending_cache: List[
            Tuple[int, List[FailedTrajectoryEntry]]
        ] = []
        # Chain/forest-mode current training step, set by the trainer via
        # `update_current_step` each training iteration. Used when reporting
        # teacher responses back to the pool for audit logging / node metadata.
        self._current_step: int = 0

    def start(self):
        """Start the background annotation thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TeacherAnnotationWorker already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="teacher-annotation-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("TeacherAnnotationWorker started.")

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("TeacherAnnotationWorker stopped.")

    def update_current_step(self, step: int) -> None:
        """Publish the current training step to the worker.

        Int write under the GIL is atomic; no lock needed. The value is used
        when reporting teacher responses back to chain/forest pools so their
        transition logs and node metadata carry a coherent step number.
        """
        self._current_step = step

    def _fetch_next_work(self) -> Optional[Tuple[str, Any]]:
        """Return one work item or None if queue is empty.

        Legacy mode returns ("legacy", FailedTrajectoryEntry).
        Chain mode   returns ("chain",  (prompt_key, [FailedTrajectoryEntry, ...])).
        Forest mode  returns ("forest", FailedPrefixEvent).
        """
        if self.chain_pool is not None:
            if not self._chain_pending_cache:
                self._chain_pending_cache = self.chain_pool.pending_teacher_requests()
            if self._chain_pending_cache:
                return ("chain", self._chain_pending_cache.pop(0))
            return None
        if self.forest_pool is not None:
            pending = self.forest_pool.pending_teacher_requests(
                max_items=1,
                current_step=self._current_step,
            )
            if pending:
                return ("forest", pending[0])
            return None
        entries = self.failed_pool.sample_and_remove(1)
        if not entries:
            return None
        return ("legacy", entries[0])

    def _run_loop(self):
        """Main entry point for the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"TeacherAnnotationWorker crashed: {e}", exc_info=True)
        finally:
            loop.close()

    async def _async_main(self):
        """Async main loop: semaphore-gated streaming annotation.

        Always keeps up to max_concurrent tasks in flight. As soon as one
        finishes, a new entry is pulled from the pool and kicked off
        immediately — no batch barrier.
        """
        sem = asyncio.Semaphore(self.max_concurrent)
        running: set = set()

        async def _run_one_legacy(entry: FailedTrajectoryEntry):
            try:
                result = await self._annotate_one(entry)
                if isinstance(result, BranchPointEntry):
                    if self.synthetic_pool is not None:
                        # Informed prior: the newly matched prefix covers
                        # prefix_frac of the remaining solution. Since
                        # teacher annotation is only triggered when the
                        # parent's k_succ == 0 (our new gate), parent_p = 0,
                        # so initial_p_hat simplifies to prefix_frac.
                        match_len = len(result.response_ids)
                        suffix_len = (
                            len(result.teacher_suffix.suffix_ids)
                            if result.teacher_suffix is not None else 0
                        )
                        denom = match_len + suffix_len
                        initial_p_hat = (match_len / denom) if denom > 0 else 0.5
                        synth = SyntheticPromptEntry(
                            augmented_prompt_ids=list(result.prompt_ids) + list(result.response_ids),
                            data_instance=result.data_instance,
                            agent_name=result.agent_name,
                            initial_p_hat=initial_p_hat,
                        )
                        self.synthetic_pool.add([synth])
                    else:
                        self.annotated_pool.add([result])
            except Exception as e:
                logger.warning(f"Teacher annotation exception: {e}")
            finally:
                sem.release()

        async def _run_one_chain(
            prompt_key: int,
            rollouts: List[FailedTrajectoryEntry],
        ):
            """Chain-mode: try each rollout; first success appends a node."""
            try:
                # Longer responses generally carry more signal for teacher's
                # first-error alignment; try them first to minimise wasted calls.
                sorted_rollouts = sorted(
                    rollouts, key=lambda r: -len(r.response_ids)
                )
                succeeded = False
                for entry in sorted_rollouts:
                    try:
                        result = await self._annotate_one(entry)
                    except Exception as e:
                        logger.warning(
                            f"Teacher annotate attempt failed (chain "
                            f"prompt_key={prompt_key}): {e}"
                        )
                        continue
                    if (
                        isinstance(result, BranchPointEntry)
                        and len(result.response_ids) > 0
                    ):
                        self.chain_pool.on_teacher_response(
                            prompt_key=prompt_key,
                            new_augmented_ids=(
                                list(result.prompt_ids) + list(result.response_ids)
                            ),
                            data_instance=result.data_instance,
                            agent_name=result.agent_name,
                            current_step=self._current_step,
                            success=True,
                        )
                        succeeded = True
                        break
                if not succeeded:
                    # All N rollouts exhausted without a valid annotation;
                    # ABANDON this chain.
                    self.chain_pool.on_teacher_response(
                        prompt_key=prompt_key,
                        new_augmented_ids=None,
                        data_instance=None,
                        agent_name="",
                        current_step=self._current_step,
                        success=False,
                    )
            except Exception as e:
                logger.warning(
                    f"Chain teacher dispatch crashed for prompt_key={prompt_key}: {e}"
                )
                # Defensive: mark chain ABANDONED so it doesn't hang in
                # DEEPENING_REQUESTED forever.
                try:
                    self.chain_pool.on_teacher_response(
                        prompt_key=prompt_key,
                        new_augmented_ids=None,
                        data_instance=None,
                        agent_name="",
                        current_step=self._current_step,
                        success=False,
                    )
                except Exception:
                    pass
            finally:
                sem.release()

        async def _run_one_forest(event: FailedPrefixEvent):
            """Forest-mode: annotate one failed rollout and insert one child."""
            try:
                result = await self._annotate_one(event.failed_entry)
                if isinstance(result, BranchPointEntry):
                    self.forest_pool.on_teacher_response(
                        event_id=event.event_id,
                        annotated_entry=result,
                        current_step=self._current_step,
                        success=True,
                    )
                else:
                    self.forest_pool.on_teacher_response(
                        event_id=event.event_id,
                        annotated_entry=None,
                        current_step=self._current_step,
                        success=False,
                    )
            except Exception as e:
                logger.warning(
                    f"Forest teacher dispatch crashed for event={event.event_id}: {e}"
                )
                try:
                    self.forest_pool.on_teacher_response(
                        event_id=event.event_id,
                        annotated_entry=None,
                        current_step=self._current_step,
                        success=False,
                    )
                except Exception:
                    pass
            finally:
                sem.release()

        # Long-lived sentinel: resolves when threading stop_event is set.
        # Used to race against sem.acquire() without wait_for, which has
        # a cancel-race permit leak in CPython < 3.12 (cpython#111693).
        async def _poll_stop():
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)

        stop_sentinel = asyncio.create_task(_poll_stop())

        # Track current acquire_task so finally can clean it up on
        # outer cancellation (issue: orphaned acquire leaks a permit).
        # Ownership rule: current_acquire is non-None IFF we hold a permit
        # that has not yet been transferred to a _run_one task or released.
        current_acquire: Optional[asyncio.Task] = None

        def _release_if_acquired(fut: asyncio.Task):
            """Sync callback: release permit if the task completed successfully."""
            if not fut.cancelled() and fut.exception() is None:
                sem.release()

        try:
            while not stop_sentinel.done():
                current_acquire = asyncio.create_task(sem.acquire())

                await asyncio.wait(
                    [current_acquire, stop_sentinel],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_sentinel.done():
                    break  # all cleanup in finally

                # Verify acquisition success. Surfaces any internal exception
                # (e.g., loop closed) immediately instead of silently breaching
                # max_concurrent. On failure, falls through to finally which
                # sees current_acquire.exception() is not None and skips release.
                current_acquire.result()

                # acquire_task completed — we hold a permit.
                # Keep current_acquire non-None until ownership is safely
                # transferred (to _run_one_*) or the permit is released.
                # Any BaseException before that point will be caught by
                # finally, which checks current_acquire and releases.
                work = self._fetch_next_work()
                if work is None:
                    sem.release()
                    current_acquire = None  # permit returned, clear tracking
                    # Race sleep against stop_sentinel for instant shutdown response.
                    sleep_task = asyncio.create_task(asyncio.sleep(self.poll_interval))
                    try:
                        await asyncio.wait(
                            [sleep_task, stop_sentinel],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not sleep_task.done():
                            sleep_task.cancel()
                    continue

                kind, payload = work
                if kind == "chain":
                    prompt_key, rollouts = payload
                    task = asyncio.create_task(
                        _run_one_chain(prompt_key, rollouts)
                    )
                elif kind == "forest":
                    task = asyncio.create_task(_run_one_forest(payload))
                else:
                    # legacy / flat synth / teacher-suffix-synthesis mode
                    task = asyncio.create_task(_run_one_legacy(payload))
                current_acquire = None  # ownership transferred

                running.add(task)
                task.add_done_callback(running.discard)

        finally:
            # Sole cleanup exit: handles normal break, outer Cancel,
            # and BaseException uniformly. Fully synchronous before
            # drain — immune to secondary Cancel.
            if current_acquire is not None:
                if not current_acquire.done():
                    current_acquire.cancel()
                    current_acquire.add_done_callback(_release_if_acquired)
                elif not current_acquire.cancelled() and current_acquire.exception() is None:
                    sem.release()

            stop_sentinel.cancel()

            # Graceful drain with cancellation safety: if we get
            # cancelled again during drain, still force-cancel everything.
            # Re-raise CancelledError afterwards to honour the asyncio
            # cancellation contract — callers must see cancellation, not
            # a silent normal return.
            if running:
                logger.info(
                    f"TeacherAnnotationWorker draining {len(running)} "
                    f"in-flight tasks..."
                )
                cancelled_during_drain = False
                try:
                    _, pending = await asyncio.wait(running, timeout=30)
                except asyncio.CancelledError:
                    pending = {t for t in running if not t.done()}
                    cancelled_during_drain = True

                if pending:
                    logger.warning(
                        f"Force-cancelling {len(pending)} timed-out tasks"
                    )
                    for t in pending:
                        t.cancel()
                    try:
                        await asyncio.wait(pending, timeout=5)
                    except asyncio.CancelledError:
                        cancelled_during_drain = True

                if cancelled_during_drain:
                    raise asyncio.CancelledError()

    async def _annotate_one(
        self, entry: FailedTrajectoryEntry
    ) -> Optional[BranchPointEntry]:
        """Annotate a single failed trajectory with teacher suffix."""
        self.total_attempted += 1

        try:
            # Create a temporary agent loop instance for this annotation.
            # This gives us access to _score_node, tokenizer, etc.
            # server_manager is None since we don't generate model tokens here.
            agent_loop = self.agent_loop_class(
                self.agent_loop_config, self.server_manager, self.tokenizer
            )

            result = await agent_loop.synthesize_teacher_suffix(
                entry=entry,
                min_prefix_match_tokens=self.min_prefix_match_tokens,
                min_prefix_match_ratio=self.min_prefix_match_ratio,
            )

            if result is not None:
                self.total_succeeded += 1
            return result

        except Exception as e:
            logger.warning(f"Teacher annotation failed for tree_id={entry.tree_id}: {e}")
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Return metrics for logging. Computes per-step deltas since last read."""
        step_attempted = self.total_attempted - self._last_read_attempted
        step_succeeded = self.total_succeeded - self._last_read_succeeded
        self._last_read_attempted = self.total_attempted
        self._last_read_succeeded = self.total_succeeded
        return {
            "teacher_worker/total_attempted": self.total_attempted,
            "teacher_worker/total_succeeded": self.total_succeeded,
            "teacher_worker/success_rate": (
                self.total_succeeded / max(self.total_attempted, 1)
            ),
            "teacher_worker/step_attempted": step_attempted,
            "teacher_worker/step_succeeded": step_succeeded,
        }
