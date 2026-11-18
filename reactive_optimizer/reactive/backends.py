"""Who answers a decision: nobody (keep), a folder of canned edits (scripted), or a coding agent (agent)."""
import glob
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

# The agent used for every reported run (see docs/method.md): a command-line coding agent run headless.
AGENT_EXECUTABLE = "claude"
AGENT_MODEL = "claude-opus-5-5"
AGENT_ENV_PREFIXES = ("CLAUDE", "ANTHROPIC_MODEL")  # dropped so a nested call starts from a clean environment


@dataclass
class Answer:
    text: str = ""
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str = ""


class KeepBackend:
    """Dry run: never asks anyone, so the starting program runs unchanged."""

    name = "keep"
    asks = False

    def ask(self, workspace: str, message: str, decision: int) -> Answer:
        return Answer()


class ScriptedBackend:
    """Testing without cost: at decision k, copies <folder>/<k>_*.py into the workspace.

    `003_updater.py` replaces updater.py at decision 3; `003_analysis.py` replaces analysis.py.
    """

    name = "scripted"
    asks = True

    def __init__(self, folder: str):
        self.folder = folder

    def ask(self, workspace: str, message: str, decision: int) -> Answer:
        done = []
        for src in sorted(glob.glob(os.path.join(self.folder, f"{decision:03d}_*.py"))):
            target = os.path.basename(src).split("_", 1)[1]
            shutil.copyfile(src, os.path.join(workspace, target))
            done.append(target)
        return Answer(text=f"scripted: replaced {done}" if done else "scripted: kept")


class AgentBackend:
    """One headless coding-agent conversation for the whole run.

    The agent gets file tools only (no shell, no network), confined to the workspace folder.
    """

    name = "agent"
    asks = True

    def __init__(self, model: str = AGENT_MODEL, effort: str = "medium", timeout: int = 900,
                 retries: int = 2, executable: str = AGENT_EXECUTABLE):
        self.model, self.effort, self.timeout, self.retries = model, effort, timeout, retries
        self.executable = executable
        self.session_id = str(uuid.uuid4())
        self.started = False

    def _command(self) -> list[str]:
        cmd = [self.executable, "-p", "--output-format", "json", "--model", self.model,
               "--effort", self.effort, "--restricted", "--strict-mcp-config",
               "--permission-mode", "acceptEdits", "--tools", "Read,Edit,Write,Glob,Grep"]
        cmd += ["--resume", self.session_id] if self.started else ["--session-id", self.session_id]
        return cmd

    def ask(self, workspace: str, message: str, decision: int) -> Answer:
        # A clean environment, so the agent behaves the same from a terminal or another session.
        env = {k: v for k, v in os.environ.items() if not k.startswith(AGENT_ENV_PREFIXES)}
        t0, last_error = time.time(), ""
        for attempt in range(self.retries + 1):
            try:
                p = subprocess.run(self._command(), input=message, text=True, capture_output=True,
                                   cwd=workspace, env=env, timeout=self.timeout)
                out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else None
                if out is not None and out.get("session_id"):
                    self.started = True  # the conversation exists, even if this turn failed
                if p.returncode == 0 and out is not None and not out.get("is_error"):
                    return Answer(text=out.get("result") or "", cost_usd=float(out.get("total_cost_usd") or 0.0),
                                  seconds=time.time() - t0)
                last_error = (out or {}).get("result") or p.stderr.strip()[-2000:] or p.stdout.strip()[-2000:]
            except subprocess.TimeoutExpired:
                last_error = f"agent timed out after {self.timeout} s"
            except (OSError, json.JSONDecodeError) as e:
                last_error = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
        return Answer(seconds=time.time() - t0, error=last_error or "agent call failed")


def make_backend(spec: str, **options):
    """'keep', 'scripted:<folder>' or 'agent'."""
    if spec == "keep":
        return KeepBackend()
    if spec.startswith("scripted:"):
        return ScriptedBackend(spec.split(":", 1)[1])
    if spec == "agent":
        return AgentBackend(**options)
    raise ValueError(f"unknown backend '{spec}'")
