"""Cooperative stop at optimizer boundaries, including Slurm pre-timeout signals."""
from contextlib import AbstractContextManager
import os
import signal
import time
from pathlib import Path


class StopController(AbstractContextManager):
    def __init__(self, minutes=0, reserve_seconds=120):
        self.requested = False
        self.reason = None
        self.handlers = {}
        self.slurm_deadline = None
        self.deadline = time.monotonic() + minutes*60 if minutes else None
        end = os.environ.get('VLM_DEADLINE_EPOCH') or os.environ.get('SLURM_JOB_END_TIME')
        if end:
            remaining = float(end) - time.time() - reserve_seconds
            slurm_deadline = time.monotonic() + remaining
            self.slurm_deadline = slurm_deadline
            self.deadline = min(self.deadline, slurm_deadline) if self.deadline else slurm_deadline

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            self.handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self.receive)
        return self

    def receive(self, number, frame):
        self.requested = True
        self.reason = signal.Signals(number).name

    def externally_stopped(self):
        sentinel = os.environ.get('VLM_STOP_FILE')
        return self.requested or (sentinel is not None and Path(sentinel).exists()) or (self.slurm_deadline is not None and time.monotonic() >= self.slurm_deadline)

    def exhausted(self):
        return self.externally_stopped() or (self.deadline is not None and time.monotonic() >= self.deadline)

    def __exit__(self, *exc):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)
