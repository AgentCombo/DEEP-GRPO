import os
import tempfile
import shutil
import subprocess
import asyncio
from typing import Any, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

import logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeExecTool(BaseTool):
    _semaphore: asyncio.Semaphore = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        max_concurrency = config.get("max_concurrency", 16)
        if CodeExecTool._semaphore is None:
            CodeExecTool._semaphore = asyncio.Semaphore(max_concurrency)

        self.base_workdir = config.get("base_workdir", "/tmp/agent_code")
        os.makedirs(self.base_workdir, exist_ok=True)

        self.timeout = config.get("timeout", 10)
        self.python_exec = config.get("python_exec", "python")
        self.init_files_base_dir = config.get("init_files_base_dir")


    async def create(self, instance_id: str = None, **kwargs) -> str:
        instance_id = instance_id or str(uuid4())
        workdir = os.path.join(self.base_workdir, instance_id)
        os.makedirs(workdir, exist_ok=True)

        # Copy files to workdir
        init_files = kwargs.get("init_files")
        if self.init_files_base_dir and os.path.exists(self.init_files_base_dir):
            if init_files:
                for file in init_files:
                    src_file = os.path.join(self.init_files_base_dir, file)
                    for file in init_files:
                        if os.path.exists(src_file):
                            shutil.copy(src_file, workdir)
                        else:
                            logger.warning(f"Init file {src_file} does not exist.")

        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        workdir = os.path.join(self.base_workdir, instance_id)

        async with CodeExecTool._semaphore:
            try:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".py", dir=workdir, delete=False
                ) as f:
                    f.write(code)
                    file_path = f.name

                    loop = asyncio.get_running_loop()
                    def blocking_call():
                        return subprocess.run(
                            [self.python_exec, file_path],
                            capture_output=True,
                            text=True,
                            cwd=workdir,
                            timeout=self.timeout,
                    )
                result = await loop.run_in_executor(None, blocking_call)

                output = result.stdout if result.returncode == 0 else result.stderr
            except subprocess.TimeoutExpired:
                output = f"⏱️ Timeout: execution exceeded {self.timeout} seconds"
            except Exception as e:
                output = f"❌ Execution error: {e}"

        return output, 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        workdir = os.path.join(self.base_workdir, instance_id)
        shutil.rmtree(workdir, ignore_errors=True)
