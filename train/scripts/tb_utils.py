"""Shared TensorBoard log loading utilities."""

import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_logs(log_dir="logs"):
    """Load TensorBoard events from log directory.

    Prefers flat event files in the given directory (direct training output).
    Falls back to subdirectories (e.g. logs/overfit/).
    Returns (path, EventAccumulator).
    """
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    dirs = sorted(glob.glob(os.path.join(log_dir, "*/")))

    if event_files:
        path = log_dir
    elif dirs:
        path = dirs[-1]
    else:
        print(f"No TensorBoard logs found in {log_dir}/")
        sys.exit(1)

    ea = EventAccumulator(path, size_guidance={"scalars": 0})  # load all
    ea.Reload()
    return path, ea


def get_latest(ea, tag):
    """Get latest ScalarEvent for a tag, or None."""
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return None
    events = ea.Scalars(tag)
    return events[-1] if events else None


def get_history(ea, tag, n=None):
    """Get scalar history, optionally last n entries."""
    scalars = ea.Tags().get("scalars", [])
    if tag not in scalars:
        return []
    events = ea.Scalars(tag)
    if n is not None:
        return events[-n:]
    return events


def get_all_at_step(ea, tags, step):
    """Get values for multiple tags at a specific step.

    Returns dict mapping tag -> value (or None if not found).
    """
    result = {}
    for tag in tags:
        events = get_history(ea, tag)
        val = None
        for e in events:
            if e.step == step:
                val = e.value
                break
        result[tag] = val
    return result
