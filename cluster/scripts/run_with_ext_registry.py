#!/usr/bin/env python
import sys, runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # .../recurrent-pretraining
sys.path.insert(0, str(REPO))

# inject our model preset
from cluster.registry.model_registry import install as _install
_install()

# hand control to the repo's train.py, preserving CLI args
train_py = REPO / "train.py"
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path(str(train_py), run_name="__main__")
