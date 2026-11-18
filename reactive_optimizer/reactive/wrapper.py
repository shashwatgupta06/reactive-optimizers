"""Reactive optimizer: an agent rewrites the optimizer program while the model trains.

Our own implementation of the idea in "Reactive Optimizers" (Bai et al., arXiv 2610.33826).
The wrapper owns two programs in a workspace folder, `updater.py` (run at every update) and
`analysis.py` (run every EVERY updates). At decision points it pauses training, sends the agent
what the run has reported, and installs whatever the agent changed, with a trial window and
rollback. Single process, single GPU, blocking decisions, no checkpoint/resume.
"""
import collections
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from .backends import Answer
from .schedule import decision_updates

TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ReactiveOptimizer:
    def __init__(self, optimizer, model, *, total_updates, updates_per_epoch, goal, run_dir,
                 backend, decisions=40, trial=20, max_cost_usd=None, facts="", evaluation="",
                 starting="", model_summary="", reference_files=(), templates="cars", spec_extra=None,
                 spec_note="", validation_fields=""):
        self.model, self.backend = model, backend
        self.total_updates, self.updates_per_epoch = total_updates, updates_per_epoch
        self.trial, self.max_cost_usd = trial, max_cost_usd
        self.named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.update = 0  # number of updates applied so far

        self.run_dir = run_dir
        self.ws = os.path.join(run_dir, "workspace")
        for d in ("inbox", "programs", "reference"):
            os.makedirs(os.path.join(self.ws, d), exist_ok=True)
        self._events = open(os.path.join(run_dir, "events.jsonl"), "a")
        self._loss_csv = open(os.path.join(self.ws, "inbox", "train_loss.csv"), "a")
        self._val_csv = open(os.path.join(self.ws, "inbox", "validation.csv"), "a")
        self._loss_csv.write("update,loss\n")
        self._val_header = None

        self.spec = {
            "total_updates": total_updates, "update": 1, "updates_per_epoch": updates_per_epoch,
            "user_optimizer": optimizer, **(spec_extra or {}),
        }
        self.task_templates = os.path.join(TEMPLATES, templates)
        self.spec_note, self.validation_fields = spec_note, validation_fields
        self._write_workspace(optimizer, goal, facts, evaluation, starting, model_summary, reference_files)

        # decisions
        self.decision_at = decision_updates(total_updates, decisions) if backend.asks else []
        self.decisions_made, self.decisions_planned = 0, len(self.decision_at)
        self.wakes_left = max(1, decisions // 2) if backend.asks else 0
        self.last_decision_update = 0
        self.cost_usd, self.wait_seconds = 0.0, 0.0
        self._wake_reason = None
        self._errors = []            # shown at the next decision
        self._new_reports = []       # analysis reports since the previous decision
        self._new_validation = []    # validation results since the previous decision
        self._validation_for_analysis = []

        # programs
        self.version = 0             # counts every program we tried to install
        self.running_version = self.prev_version = 0
        self._attempted = {}         # file -> sha of the last version we tried to install
        self.updater = self.prev_updater = self.prev_state = None
        self.installed_at = 1
        self._snaps = collections.deque(maxlen=2)  # (update, model state) known to be usable
        self.analysis_mod, self.analysis_memory = None, {}
        self._losses, self._losses_since_decision = [], []
        self._step_gaps, self._last_step_time = [], None
        self.ms_per_update = self.ms_per_update_start = None
        self.swaps = self.rollbacks = 0

        self._install_updater(first=True)
        self._load_analysis(first=True)

    # ------------------------------------------------------------------ workspace
    def _write_workspace(self, optimizer, goal, facts, evaluation, starting, model_summary, reference_files):
        for name in ("updater.py", "analysis.py"):
            shutil.copyfile(os.path.join(self.task_templates, name), os.path.join(self.ws, name))
        shutil.copyfile(os.path.join(self.task_templates, "example_updater.py"),
                        os.path.join(self.ws, "reference", "example_updater.py"))
        for src in reference_files:
            shutil.copyfile(src, os.path.join(self.ws, "reference", os.path.basename(src)))
        group_of = {id(p): i for i, g in enumerate(optimizer.param_groups) for p in g["params"]}
        with open(os.path.join(self.ws, "reference", "starting_optimizer.txt"), "w") as f:
            f.write(f"{type(optimizer).__name__}\n")
            for i, g in enumerate(optimizer.param_groups):
                hp = {k: v for k, v in g.items() if k != "params"}
                f.write(f"group {i}: {len(g['params'])} tensors, {hp}\n")
            f.write(f"\nschedule: {starting}\n")
        table = "\n".join(f"{n}  {tuple(p.shape)}  group {group_of.get(id(p), '-')}" for n, p in self.named_params)
        text = open(os.path.join(TEMPLATES, "INSTRUCTIONS.md")).read()
        for key, value in {"GOAL": goal, "FACTS": facts, "EVALUATION": evaluation, "STARTING": starting,
                           "MODEL": model_summary, "PARAM_TABLE": table, "TRIAL": str(self.trial),
                           "SNAP": "100", "SPEC_NOTE": self.spec_note,
                           "VALIDATION_FIELDS": self.validation_fields}.items():
            text = text.replace("{{" + key + "}}", value)
        with open(os.path.join(self.ws, "INSTRUCTIONS.md"), "w") as f:
            f.write(text)
        self.instructions = text
        open(os.path.join(self.ws, "NOTES.md"), "a").close()
    def _event(self, kind, **fields):
        self._events.write(json.dumps({"event": kind, "update": self.update + 1, "time": time.time(), **fields}) + "\n")
        self._events.flush()

    # ------------------------------------------------------------------ programs
    def _import(self, filename):
        """Import a workspace file under a fresh module name, with workspace helpers reloaded."""
        ws = os.path.realpath(self.ws)
        for name, mod in list(sys.modules.items()):
            path = getattr(mod, "__file__", None)
            if path and os.path.dirname(os.path.realpath(path)) == ws:
                del sys.modules[name]
        self._import_n = getattr(self, "_import_n", 0) + 1
        spec = importlib.util.spec_from_file_location(
            f"_ro_ws_{filename[:-3]}_{self._import_n}", os.path.join(self.ws, filename))
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, self.ws)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(self.ws)
        return mod

    def _program_hashes(self):
        files = sorted(f for f in os.listdir(self.ws) if f.endswith(".py"))
        return {f: _sha(os.path.join(self.ws, f)) for f in files}

    def _archive(self, tag):
        for f in self._program_hashes():
            shutil.copyfile(os.path.join(self.ws, f), os.path.join(self.ws, "programs", f"{self.version:03d}_{tag}_{f}"))

    def _model_state(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def _install_updater(self, first=False):
        """Build updater.py. Returns True if a new updater took over."""
        self.spec["update"] = self.update + 1
        try:
            mod = self._import("updater.py")
            new = mod.build(self.named_params, previous=self.updater, spec=self.spec)
            for attr in ("step", "state_dict", "load_state_dict"):
                if not callable(getattr(new, attr, None)):
                    raise TypeError(f"the object returned by build() has no method {attr}()")
            new.state_dict()
        except Exception:
            if first:
                raise
            self.version += 1
            self._archive("rejected")
            self._errors.append("updater.py was NOT installed (import or build failed); the previous "
                                "program keeps running.\n" + traceback.format_exc(limit=6))
            self._event("build_failed")
            return False
        if not first:
            self.version += 1
            self.swaps += 1
            self.prev_updater = self.updater
            self.prev_state = copy.deepcopy(self.updater.state_dict())
            self.prev_version = self.running_version
            self._snaps.clear()
        self._snaps.append((self.update, self._model_state()))
        self.updater, self.installed_at, self.running_version = new, self.update + 1, self.version
        self._archive("installed")
        self._event("installed", version=self.version)
        return True

    def _load_analysis(self, first=False):
        try:
            mod = self._import("analysis.py")
            if not callable(getattr(mod, "analyze", None)):
                raise TypeError("analysis.py defines no analyze(obs, memory)")
            every = int(getattr(mod, "EVERY", 50))
            if every < 1:
                raise ValueError("EVERY must be a positive int")
        except Exception:
            if first:
                raise
            self._errors.append("analysis.py was NOT reloaded; the previous analysis keeps running.\n"
                                + traceback.format_exc(limit=6))
            return
        self.analysis_mod, self.analysis_every = mod, every

    def _revert(self, reason):
        """Something went wrong under the running updater: restore weights and the previous updater."""
        if self.prev_updater is None:
            raise RuntimeError(f"the starting program failed at update {self.update + 1}: {reason}")
        snap_update, state = self._snaps[0]
        self.model.load_state_dict(state)
        failed_version, self.running_version = self.running_version, self.prev_version
        self.updater, self.prev_updater = self.prev_updater, None
        self.updater.load_state_dict(self.prev_state)
        self.prev_state = None
        self.rollbacks += 1
        self._snaps.clear()
        self._snaps.append((self.update, self._model_state()))
        age = self.update + 1 - self.installed_at
        self._errors.append(
            f"Program version {failed_version} (installed at update {self.installed_at}) FAILED at update "
            f"{self.update + 1}, {age} updates after it took over. The weights were restored to their values "
            f"after update {snap_update} and the previous updater (version {self.running_version}) is running again with its state restored. "
            f"updater.py in your folder is still the failed version.\n{reason}")
        self.installed_at = self.update + 1
        self._event("rollback", failed_version=failed_version, restored_to=snap_update)
        self._wake_reason = self._wake_reason or "your updater failed and was rolled back"

    # ------------------------------------------------------------------ analysis
    def _run_analysis(self, u):
        if self._step_gaps:  # the median ignores evaluations and decisions between updates
            self.ms_per_update = 1000 * sorted(self._step_gaps)[len(self._step_gaps) // 2]
            if self.ms_per_update_start is None and self.swaps == 0:
                self.ms_per_update_start = self.ms_per_update
        self._step_gaps = []
        obs = SimpleNamespace(
            update=u, total_updates=self.total_updates, updates_per_epoch=self.updates_per_epoch,
            losses=self._losses, params=dict(self.named_params),
            grads={n_: p.grad for n_, p in self.named_params}, updater=self.updater,
            validation=self._validation_for_analysis, wall_ms_per_update=self.ms_per_update)
        try:
            with torch.no_grad():
                out = self.analysis_mod.analyze(obs, self.analysis_memory)
            if out:
                if "report" in out:
                    line = json.dumps({"update": u, "report": out["report"]})
                    with open(os.path.join(self.ws, "inbox", "reports.jsonl"), "a") as f:
                        f.write(line + "\n")
                    self._new_reports.append(line)
                if out.get("wake"):
                    self._wake_reason = self._wake_reason or f"your analysis asked: {out['wake']}"
        except Exception:
            self._errors.append(f"analysis.py raised at update {u}:\n" + traceback.format_exc(limit=6))
        self._losses, self._validation_for_analysis = [], []
        # A snapshot only counts as usable once the run has trained on from it without trouble.
        self._snaps.append((u - 1, self._model_state()))

    # ------------------------------------------------------------------ decisions
    def _decision_due(self, u):
        if not self.backend.asks or (self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd):
            return None
        if self.decision_at and u >= self.decision_at[0]:
            while self.decision_at and u >= self.decision_at[0]:
                self.decision_at.pop(0)
            return "scheduled"
        if self._wake_reason and self.wakes_left > 0 and u - self.last_decision_update >= 100:
            self.wakes_left -= 1
            return self._wake_reason
        return None

    def _message(self, u, reason):
        k = self.decisions_made
        ls = self._losses_since_decision
        lines = [f"# Decision {k} at update {u} of {self.total_updates} "
                 f"(epoch {(u - 1) / self.updates_per_epoch:.2f} of {self.total_updates / self.updates_per_epoch:.0f})",
                 f"Why now: {reason}.",
                 f"Running program: version {self.running_version} (programs/{self.running_version:03d}_installed_updater.py), "
                 f"in effect since update {self.installed_at}."]
        if self._errors:
            lines += ["", "## Errors since your previous decision"] + self._errors
        lines += ["", f"## Training loss since your previous decision ({len(ls)} updates)"]
        if ls:
            chunks = min(10, len(ls))
            size = len(ls) / chunks
            for c in range(chunks):
                part = ls[round(c * size): round((c + 1) * size)]
                first = u - len(ls) + round(c * size)
                lines.append(f"- updates {first}-{first + len(part) - 1}: mean {sum(part) / len(part):.4f}")
        lines += ["", "## Validation since your previous decision"]
        fmt = lambda v: ", ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}"
                                  for k, x in v.items() if k not in ("update", "epoch"))
        lines += [f"- update {v['update']} (epoch {v['epoch']}): {fmt(v)}"
                  for v in self._new_validation] or ["- none"]
        lines += ["", f"## Analysis reports since your previous decision ({len(self._new_reports)}; all in inbox/reports.jsonl)"]
        shown = self._new_reports[-12:]
        lines += [r if len(r) <= 1500 else r[:1500] + " ...(truncated)" for r in shown] or ["- none"]
        if self.ms_per_update is not None:
            start = f"{self.ms_per_update_start:.1f} ms" if self.ms_per_update_start else "not measured"
            lines += ["", f"Time per update: {self.ms_per_update:.1f} ms now; starting program: {start}."]
        nxt = self.decision_at[0] if self.decision_at else None
        lines += [f"Scheduled decisions left after this one: {len(self.decision_at)}"
                  + (f"; the next is at update {nxt}." if nxt else ". This is the last scheduled decision."),
                  "", "Protocol and interfaces are in INSTRUCTIONS.md. Edit updater.py and/or analysis.py, or keep "
                  "them, then end with your rationale (Observed / Changed or kept / Prediction)."]
        return "\n".join(lines)

    def _decide(self, u, reason):
        self.decisions_made += 1
        k = self.decisions_made
        message = self._message(u, reason)
        with open(os.path.join(self.ws, "inbox", f"decision_{k:03d}.md"), "w") as f:
            f.write(message)
        if k == 1:
            message = self.instructions + "\n\n---\n\n" + message
        before = self._program_hashes()
        answer: Answer = self.backend.ask(self.ws, message, k)
        self.cost_usd += answer.cost_usd
        self.wait_seconds += answer.seconds
        after = self._program_hashes()
        changed = sorted(f for f in after if after[f] != before.get(f) and after[f] != self._attempted.get(f))
        with open(os.path.join(self.ws, "inbox", f"decision_{k:03d}_answer.md"), "w") as f:
            f.write(answer.text or f"(no answer: {answer.error})")
        self._errors, self._new_reports, self._new_validation = [], [], []
        self._losses_since_decision, self._wake_reason = [], None
        self.last_decision_update = u
        if answer.error:
            self._errors.append(f"Your previous turn failed ({answer.error}); nothing was changed.")
            changed = []
        self._attempted.update({f: after[f] for f in changed})
        installed = False
        if any(f != "analysis.py" for f in changed):
            installed = self._install_updater()
        if "analysis.py" in changed or (changed and not installed):
            self._load_analysis()
        self._event("decision", decision=k, reason=reason, changed=changed, installed=installed,
                    cost_usd=answer.cost_usd, seconds=answer.seconds, error=answer.error)
        print(f"[reactive] decision {k} at update {u}: changed {changed or 'nothing'}"
              f"{' (updater installed)' if installed else ''}; {answer.seconds:.0f} s, "
              f"${self.cost_usd:.2f} so far", flush=True)
        self._last_step_time = None  # do not count the wait as training time

    # ------------------------------------------------------------------ training interface
    def step(self, loss):
        """Call after loss.backward(). Applies update number self.update + 1."""
        u = self.update + 1
        value = float(loss.detach())
        now = time.time()
        if self._last_step_time is not None:
            self._step_gaps.append(now - self._last_step_time)
        self._last_step_time = now
        self._loss_csv.write(f"{u},{value:.6f}\n")
        self._losses.append(value)
        self._losses_since_decision.append(value)

        if not math.isfinite(value):
            self._revert("The training loss became non-finite.")
            self.update = u  # this update is skipped: its gradients are not usable
            return
        if u % self.analysis_every == 0:
            self._run_analysis(u)
        reason = self._decision_due(u)
        if reason:
            self._loss_csv.flush()
            self._decide(u, reason)

        grads = {n: p.grad for n, p in self.named_params}
        ctx = SimpleNamespace(loss=loss.detach(), update=u, total_updates=self.total_updates)
        try:
            self.updater.step(grads, u, ctx)
        except Exception:
            self._revert("updater.step raised:\n" + traceback.format_exc(limit=6))
            self.updater.step(grads, u, ctx)
        self.update = u

    def report_validation(self, epoch, **metrics):
        """Validation-set results; the agent sees these. Never pass test-set values here."""
        v = {"update": self.update, "epoch": epoch, **{k: float(x) for k, x in metrics.items()}}
        self._new_validation.append(v)
        self._validation_for_analysis.append(v)
        if self._val_header is None:
            self._val_header = list(v)
            self._val_csv.write(",".join(self._val_header) + "\n")
        self._val_csv.write(",".join(f"{v.get(k, '')}" for k in self._val_header) + "\n")
        self._val_csv.flush()

    def current_lrs(self):
        groups = getattr(self.updater, "param_groups", None) or []
        return [float(g["lr"]) for g in groups if "lr" in g]

    def summary(self):
        return {"decisions": self.decisions_made, "decisions_planned": self.decisions_planned,
                "swaps": self.swaps, "rollbacks": self.rollbacks, "program_version": self.running_version,
                "agent_cost_usd": round(self.cost_usd, 4), "agent_wait_seconds": round(self.wait_seconds, 1)}

    def close(self):
        self._event("close", **self.summary())
        for f in (self._events, self._loss_csv, self._val_csv):
            f.close()
