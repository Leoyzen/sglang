# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Capture-audit guard for DCP-era prefill CUDA graph capture.

Openspec change ``enable-dcp-bcg-prefill-cudagraph`` task 1.1.

Why this exists
---------------
The design (D2) keeps DCP attention kernels and their KV all-gather /
LSE-reduce collectives OUTSIDE captured prefill graph segments: symmetric
memory registration during capture is rank-divergence-prone (#34230) and
vLLM #36070 shows full-graph capture baking in stale DCP state. Until the
metadata plumbing (tasks 2.x/3.x) lands and attention is provably an eager
break, nothing mechanically verifies that a stray NCCL/symm-mem collective
did not get recorded into a captured segment. This helper provides that
verification in debug mode: it wraps the torch.distributed collective
entry points, records every invocation that happens while a graph segment
is being captured, and fails loudly when one is observed between capture
start and end.

It is strictly a debug utility — production capture is untouched. The
instrumentation is armed only when ``SGLANG_DEBUG_CAPTURE_COLLECTIVE_AUDIT``
is set (or a debug flag param is passed) and imposes Python-level overhead
on every distributed call, so it must never run by default.

Design notes
------------
* Wrapping happens at the ``torch.distributed`` module attribute level,
  which is where SGlang's own code and the collectives reached through
  helper modules resolve their entry points. A re-entrant guard flag keeps
  wrapped calls re-entrant-safe (a collective that internally calls another
  wrapped entry point records once).
* The recorder only counts when ``torch.cuda.is_current_stream_capturing()``
  reports an active capture, so idle-time collectives outside capture
  windows never pollute the audit.
* Failures raise :class:`CollectiveCaptureAuditError` (a RuntimeError
  subclass) naming the offending op, so capture aborts instead of shipping
  a graph with the collective baked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# torch.distributed entry points audited as DCP-relevant collectives.
# Anything recorded inside a captured segment of these means a DCP
# communication got baked into graph memory.
_AUDITED_COLLECTIVES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "gather",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "reduce",
    "broadcast",
    "scatter",
    "send",
    "recv",
    "barrier",
    "all_to_all",
    "all_to_all_single",
)

# Attribute on torch.distributed holding the original function per name.
_ORIGINALS_ATTR = "_sglang_capture_audit_originals"


class CollectiveCaptureAuditError(RuntimeError):
    """Raised when a collective is invoked inside a captured graph segment."""


@dataclass
class _AuditState:
    """Mutable state of one active audit session."""

    capture_stack: int = 0
    # (op_name, stack summary) pairs recorded during capture.
    recorded: List[Dict[str, object]] = field(default_factory=list)
    # Thread-reentrancy guard: num times current thread entered a wrapper.
    _depth: int = 0


class CaptureCollectiveAuditor:
    """Wrap ``torch.distributed`` collectives and detect in-capture calls.

    Usage (debug capture path)::

        auditor = CaptureCollectiveAuditor()
        with auditor.collective_capture_scope("prefill-bucket-16"):
            ... run the capture forward ...
        auditor.assert_clean()
    """

    def __init__(self) -> None:
        self._state = _AuditState()
        self._installed = False

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------
    def install(self) -> None:
        """Swap the audited ``torch.distributed`` entry points for wrappers."""
        if self._installed:
            return
        import torch.distributed as dist

        originals: Dict[str, Callable] = {}
        for name in _AUDITED_COLLECTIVES:
            fn = getattr(dist, name, None)
            if not callable(fn):
                continue
            originals[name] = fn
            setattr(dist, name, self._wrap(name, fn))
        setattr(dist, _ORIGINALS_ATTR, originals)
        self._installed = True
        logger.debug("Capture collective audit: wrapping %s", sorted(originals))

    def uninstall(self) -> None:
        """Restore the original entry points (idempotent)."""
        if not self._installed:
            return
        import torch.distributed as dist

        originals = getattr(dist, _ORIGINALS_ATTR, {})
        for name, fn in originals.items():
            setattr(dist, name, fn)
        if hasattr(dist, _ORIGINALS_ATTR):
            delattr(dist, _ORIGINALS_ATTR)
        self._installed = False

    def _wrap(self, name: str, fn: Callable) -> Callable:
        state = self._state

        def wrapper(*args, **kwargs):
            capturing = torch.cuda.is_available() and (
                torch.cuda.is_current_stream_capturing()
            )
            if capturing and not state._depth:
                state.recorded.append(
                    {
                        "op": name,
                        "capture_depth": state.capture_stack,
                    }
                )
            state._depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                state._depth -= 1

        # Mirror identity metadata so tracebacks/readers see the audited op.
        fn_name = getattr(fn, "__name__", None)
        if isinstance(fn_name, str):
            object.__setattr__(wrapper, "__name__", fn_name)
        doc = getattr(fn, "__doc__", None)
        if isinstance(doc, str):
            object.__setattr__(wrapper, "__doc__", doc)
        return wrapper

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------
    def begin_capture_scope(self, label: str) -> None:
        """Mark the beginning of one captured graph segment."""
        if not self._installed:
            self.install()
        self._state.capture_stack += 1
        self._last_label = label

    def end_capture_scope(self) -> None:
        """Mark the end of one captured graph segment."""
        if self._state.capture_stack > 0:
            self._state.capture_stack -= 1

    def reset(self) -> None:
        """Forget all recorded invocations (between capture scopes)."""
        self._state.recorded.clear()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def recorded_collectives(self) -> List[Dict[str, object]]:
        """The invocations seen inside a capture scope so far."""
        return list(self._state.recorded)

    def assert_clean(self, context: Optional[str] = None) -> None:
        """Raise if any collective was recorded inside a capture scope."""
        if not self._state.recorded:
            return
        ops = ", ".join(sorted({str(r["op"]) for r in self._state.recorded}))
        where = f" in {context}" if context else ""
        raise CollectiveCaptureAuditError(
            "Captured CUDA graph segment contains NCCL/symm-mem collective "
            f"call(s){where}: [{ops}]. DCP attention collectives must run as "
            "eager break points (openspec enable-dcp-bcg-prefill-cudagraph: "
            "'No collective captured in graph'). Refusing to ship a graph "
            "with baked-in collectives."
        )


_ACTIVE_AUDITOR: Optional[CaptureCollectiveAuditor] = None


def capture_collective_audit_enabled(debug_flag: Optional[bool] = None) -> bool:
    """Whether the audit is requested for this run.

    Armed when the ``SGLANG_DEBUG_CAPTURE_COLLECTIVE_AUDIT`` env var is set
    truthy, or the caller passes ``debug_flag=True``.
    """
    if debug_flag:
        return True
    return bool(envs.SGLANG_DEBUG_CAPTURE_COLLECTIVE_AUDIT.get())


def get_active_capture_collective_auditor() -> Optional[CaptureCollectiveAuditor]:
    """The auditor installed for this process, if any."""
    return _ACTIVE_AUDITOR


def instrumented_capture_scope(label: str, debug_flag: Optional[bool] = None):
    """Context manager to wrap one capture segment when the audit is enabled.

    When disabled this is a no-op passthrough. When enabled it installs (once
    per process), opens a capture scope for ``label``, and on exit asserts no
    collective was recorded — raising :class:`CollectiveCaptureAuditError` if
    one was, so capture aborts loudly.

    Debug-only: the wrappers add a Python frame + one CUDA query per
    distributed call, which is unacceptable on the serving path.
    """
    import contextlib

    if not capture_collective_audit_enabled(debug_flag):
        return contextlib.nullcontext()

    global _ACTIVE_AUDITOR
    if _ACTIVE_AUDITOR is None:
        _ACTIVE_AUDITOR = CaptureCollectiveAuditor()
        _ACTIVE_AUDITOR.install()
    auditor = _ACTIVE_AUDITOR

    @contextlib.contextmanager
    def _scope():
        auditor.begin_capture_scope(label)
        auditor.reset()
        try:
            yield auditor
        finally:
            auditor.end_capture_scope()

    return _scope()


def audit_active_capture_segment(context: str) -> None:
    """Fail loudly if the just-captured segment recorded a collective.

    Convenience for capture loops that scope segments manually: call once
    per captured segment with a bucket/label description. Enabled when an
    auditor is active for this process (i.e. a scope was opened) — it does
    not re-read the env var, so the enablement decision lives in one place
    (:func:`instrumented_capture_scope` / :func:`CaptureCollectiveAuditor`).
    """
    auditor = get_active_capture_collective_auditor()
    if auditor is None:
        return
    try:
        auditor.assert_clean(context)
    finally:
        auditor.reset()
