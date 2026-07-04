from typing import Optional

from pathlib import Path
import asyncio
import sys
import os
import resource


semaphore = asyncio.Semaphore(128)


class CodeExecutor:
    def __init__(self, max_tool_response_length: int = 4096, default_timeout: float = 60, per_process_memory_gb: int = 8):
        self.max_tool_response_length = max_tool_response_length
        self.default_timeout = default_timeout
        self.per_process_memory_gb = per_process_memory_gb

        base_env = os.environ.copy()
        base_env.setdefault("JOBLIB_MULTIPROCESSING", "0")
        base_env.setdefault("OMP_NUM_THREADS", "1")
        base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        base_env.setdefault("MKL_NUM_THREADS", "1")
        base_env.setdefault("NUMEXPR_NUM_THREADS", "1")
        self._safe_env = base_env

    def _limit_memory_preexec(self):
        max_bytes = self.per_process_memory_gb * 1024**3
        def _set_limits():
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        return _set_limits

    def _trim_output(self, text: str) -> str:
        if len(text) <= self.max_tool_response_length:
            return text
        marker = "\n...\n[output truncated]\n...\n"
        if self.max_tool_response_length <= len(marker):
            return text[:self.max_tool_response_length]
        keep = self.max_tool_response_length - len(marker)
        head = keep // 2
        tail = keep - head
        return text[:head] + marker + text[-tail:]

    async def execute_code(
        self,
        code_str: str,
        timeout: Optional[float] = None,
        workspace: Optional[str] = None,
    ) -> str:
        # https://stackoverflow.com/questions/42639984/python3-asyncio-wait-for-communicate-with-timeout-how-to-get-partial-resul

        effective_timeout = timeout or self.default_timeout

        if workspace:
            try:
                Path(workspace).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return self._trim_output(f"[Error]: Failed to create workspace '{workspace}': {e}")

        async with semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", code_str,
                    cwd=workspace or None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self._safe_env,
                    preexec_fn=self._limit_memory_preexec()
                )
            except Exception as e:
                return self._trim_output(f"[Error]: Failed to start subprocess: {e}")

            comm_task = asyncio.create_task(proc.communicate())

            done, _ = await asyncio.wait({comm_task}, timeout=effective_timeout)

            if comm_task in done:
                stdout, _ = comm_task.result()
                output = stdout.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    output = f"[Error]: {output or f'subprocess exited with code {proc.returncode}'}"
                return self._trim_output(output)
            else:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    stdout, _ = await asyncio.wait_for(comm_task, timeout=5.0)
                    output = stdout.decode("utf-8", errors="replace")
                    msg = f"[Error]: Code execution timed out after {effective_timeout} seconds."
                    output = (output + "\n" + msg) if output else msg
                    return self._trim_output(output)
                except asyncio.TimeoutError:
                    msg = (
                        f"[Error]: Code execution timed out after {effective_timeout} seconds "
                        f"and failed to terminate within 5 seconds after kill()."
                    )
                    return self._trim_output(msg)
                except Exception as e:
                    msg = (
                        f"[Error]: Code execution timed out after {effective_timeout} seconds "
                        f"and failed to collect output: {e}"
                    )
                    return self._trim_output(msg)