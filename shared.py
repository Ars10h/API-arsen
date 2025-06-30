# shared.py

import threading

# Shared object to track training progress
progress = {
    "current_round": 0,
    "total_rounds": 0,
    "status": "idle",
    "accuracy": [],
    "mia_auc": [],
    "message": ""
}

progress_lock = threading.Lock()

def update_progress(key, value):
    with progress_lock:
        progress[key] = value

def append_metric(metric_name, value):
    with progress_lock:
        progress[metric_name].append(value)

def get_progress():
    with progress_lock:
        return progress.copy()

def get_metrics():
    with progress_lock:
        return {
            "accuracy": progress["accuracy"],
            "mia_auc": progress["mia_auc"]
        }