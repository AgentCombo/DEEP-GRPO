from __future__ import annotations

import os
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple
from uuid import uuid4
from contextlib import asynccontextmanager

from jupyter_client import AsyncKernelManager, AsyncKernelClient

import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

from .base_tool import BaseTool

__all__ = ["NotebookTool"]


class _InstanceState:
    def __init__(
        self,
        instance_id: str,
        km: AsyncKernelManager,
        kc: AsyncKernelClient,
        session_dir: str,
        stderr_file: Any = None,
        stdout_file: Any = None,
    ) -> None:
        self.instance_id = instance_id
        self.km = km
        self.kc = kc
        self.session_dir = session_dir
        self.stderr_file = stderr_file
        self.stdout_file = stdout_file
        self._exec_lock = asyncio.Lock()

    async def execute_code(self, code: str, timeout: int = 60) -> str:
        async with self._exec_lock:
            msg_id = self.kc.execute(
                code,
                store_history=True,
                stop_on_error=True,
                allow_stdin=False,
            )

            outputs: list[str] = []

            async def read_shell():
                try:
                    while True:
                        msg = await self.kc.get_shell_msg()
                        if msg.get("parent_header", {}).get("msg_id") != msg_id:
                            continue
                        if msg.get("msg_type") == "execute_reply":
                            break
                except Exception as e:
                    raise RuntimeError(f"Error reading shell message: {e}") from e

            async def read_iopub():
                try:
                    while True:
                        msg = await self.kc.get_iopub_msg()
                        if msg.get("parent_header", {}).get("msg_id") != msg_id:
                            continue
                        mtype = msg.get("msg_type")
                        content = msg.get("content", {})
                        if mtype == "stream":
                            outputs.append(content.get("text", ""))
                        elif mtype in {"execute_result", "display_data"}:
                            outputs.append(content.get("data", {}).get("text/plain", ""))
                        elif mtype == "error":
                           outputs.append(
                                f"{content.get('ename', 'Error')}: {content.get('evalue', '')}\n"
                                + "\n".join(content.get('traceback', []))
                            )
                        elif mtype == "status" and content.get("execution_state", "") == "idle":
                             break
                except Exception as e:
                    raise RuntimeError(f"Error reading iopub messages: {e}") from e

            shell_task = asyncio.create_task(read_shell())
            iopub_task = asyncio.create_task(read_iopub())

            try:
                await asyncio.wait_for(asyncio.gather(shell_task, iopub_task), timeout=timeout)
            except asyncio.TimeoutError as e:
                shell_task.cancel()
                iopub_task.cancel()

                logger.warning(f"Execution timed out for session {self.instance_id}. Interrupting kernel.")
                try:
                    await asyncio.wait_for(self.km.interrupt_kernel(), timeout=3.0)
                    logger.info(f"Successfully interrupted kernel for session {self.instance_id}.")
                except Exception as e:
                    logger.error(f"Failed to interrupt kernel for {self.instance_id}: {e}. The session might be unstable.")
                
                partial_output = "".join(outputs).strip()
                raise TimeoutError(
                    f"Executing timeout after {timeout} seconds.\n"
                    f"Partial output:\n{partial_output}"
                ) from e
            # Other exceptions are handled in the outside

            return "".join(outputs).strip()
                

class NotebookTool(BaseTool):
    _sessions: dict[str, _InstanceState] = {}
    _lock = asyncio.Lock()
    
    _max_sessions: int = int(os.getenv("NOTEBOOK_MAX_SESSIONS", 8))
    _sema = asyncio.BoundedSemaphore(_max_sessions)

    async def create(self, **kwargs) -> str:
        await self._sema.acquire()
        workdir = None
        stderr_file = None
        stdout_file = None
        km = None
        kc = None
        
        try:
            instance_id = str(uuid4())

            cfg = self.config
            kernel_name = cfg.get("kernel_name", "python3")
            init_path: Optional[str | Path] = cfg.get("init_files_dir")
            init_files = kwargs.get("init_files", None)

            base_temp_dir = tempfile.gettempdir()
            session_dir = os.path.join(base_temp_dir, f"rollout_{instance_id}")

            if os.path.exists(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)

            workdir = os.path.join(session_dir, "work")
            connection_dir = os.path.join(session_dir, "conn")
            logs_dir = os.path.join(session_dir, "logs")

            os.makedirs(workdir, exist_ok=True)
            os.makedirs(connection_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)

            # Copy files to workdir
            if init_path and os.path.exists(init_path):
                if init_files:
                    for file in init_files:
                        src_file = os.path.join(init_path, file)
                        if os.path.exists(src_file):
                            shutil.copy(src_file, workdir)
                        else:
                            logger.warning(f"Init file {src_file} does not exist.")

            cf = os.path.join(connection_dir, "kernel_connection.json")

            stderr_file = open(os.path.join(logs_dir, "kernel.err"), "wb")
            stdout_file = open(os.path.join(logs_dir, "kernel.out"), "wb")

            km = AsyncKernelManager(
                kernel_name=kernel_name,
                transport="ipc",
                connection_file=cf,
            )
            try:
                await asyncio.wait_for(
                    km.start_kernel(cwd=workdir, stderr=stderr_file, stdout=stdout_file),
                    timeout=30
                )
            except asyncio.TimeoutError as e:
                raise RuntimeError(f"The kernel process for session {instance_id} failed to launch in time.") from e

            kc = km.client()
            kc.start_channels()
            
            try:
                await asyncio.wait_for(kc.wait_for_ready(), timeout=30)
            except asyncio.TimeoutError as e:
                raise RuntimeError(f"The kernel for session {instance_id} started but failed to become ready in time.") from e

            st = _InstanceState(
                instance_id=instance_id,
                km=km,
                kc=kc,
                session_dir=session_dir,
                stderr_file=stderr_file,
                stdout_file=stdout_file,
            )
            
            async with self._lock:
                self._sessions[instance_id] = st
                
            return instance_id

        except Exception as e:
            self._sema.release()
            await self._cleanup_resources(
                km, kc, stderr_file, stdout_file, session_dir
            )

            raise 

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **_
    ) -> Tuple[str, float, dict]:
        async with self._lock:
            st = self._sessions.get(instance_id)
            
        if st is None:
            raise KeyError(f"No active session for {instance_id}")

        code = parameters.get("code")

        if code is None:
            return "", 0.0, {}
        
        if not isinstance(code, str):
            raise TypeError("parameters['code'] must be str")

        try:
            timeout = int(parameters.get("timeout", 10))
            output = await st.execute_code(code, timeout=timeout)
            return output, 0.0, {}
            
        except Exception as e:
            raise RuntimeError(f"Error executing code: {e}") from e

    async def calc_reward(self, instance_id: str, **_) -> float:
        return 0.0

    async def release(self, instance_id: str, **_):
        st = None
        async with self._lock:
            if instance_id in self._sessions:
                st = self._sessions.pop(instance_id)
        
        if st:
            await self._cleanup_resources(
                km=st.km,
                kc=st.kc,
                stderr_file=st.stderr_file,
                stdout_file=st.stdout_file,
                session_dir=st.session_dir,
            )
            self._sema.release()

    async def _cleanup_resources(
        self,
        km: Optional[AsyncKernelManager] = None,
        kc: Optional[AsyncKernelClient] = None,
        stderr_file: Optional[Any] = None,
        stdout_file: Optional[Any] = None,
        session_dir: Optional[str] = None,
    ):
        for f in [stderr_file, stdout_file]:
            if f:
                try: f.close()
                except Exception: pass

        if kc:
            try: kc.stop_channels()
            except Exception as e: logger.error(f"Failed to stop channels: {e}")

        if km:
            try: await asyncio.wait_for(km.shutdown_kernel(now=True), timeout=5)
            except Exception as e: logger.error(f"Failed to shutdown kernel: {e}")
        
        if session_dir and os.path.exists(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)