# training_main.py

from training_plain import run as train_plain
from training_dropout import run as train_dropout
from training_dp import run as train_dp

def run_federated_training(defense: str = "none"):
    if defense == "dropout":
        print("[INFO] Starting training with Dropout defense")
        train_dropout()
    elif defense == "dp":
        print("[INFO] Starting training with Differential Privacy")
        train_dp()
    else:
        print("[INFO] Starting training with NO defense")
        train_plain()

