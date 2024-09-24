"""Shared types for interacting with the execution server.

NOTE: this file gets copied by the Dockerfile into image that runs
the execution server."""
from typing import List, Tuple
from pydantic import BaseModel


class ExecuteCodeRequest(BaseModel):
    code: str
    input_expected_output_pairs: List[Tuple[str, str]]
    timeout: float
    memory_limit_bytes: int


class ExecuteCodeResult(BaseModel):
    correct: bool
