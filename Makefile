# Development entry points. Everything runs through uv (no manual venvs).
.PHONY: setup test typecheck lint download prepare status training evaluate

setup:  ## create/update the uv environment (incl. data-prep extras and dev tools)
	uv sync --all-extras
	uv run python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

test:  ## run the whole test suite
	uv run pytest

typecheck:  ## static typing: mypy and basedpyright over all python files
	uv run mypy .
	uv run basedpyright

lint:  ## ruff
	uv run ruff check .

# Run targets take the config as a positional argument: `make download config/datasets/<name>.yaml`;
# `evaluate` takes a checkpoint: `make evaluate outputs/<run>/checkpoints/<file>.pth`.
CONFIG = $(filter %.yaml %.yml,$(MAKECMDGOALS))
CHECKPOINT = $(filter %.pth,$(MAKECMDGOALS))
require_config = $(if $(CONFIG),,$(error usage: make $@ <path to a .yaml config>))
require_checkpoint = $(if $(CHECKPOINT),,$(error usage: make $@ <path to a .pth checkpoint>))

download:  ## download only (tokenizer + raw shards): make download config/datasets/<name>.yaml
	$(require_config)
	uv run python data_preparation/prepare.py download --dataset_config $(CONFIG)

prepare:  ## download and build a dataset: make prepare config/datasets/<name>.yaml
	$(require_config)
	uv run python data_preparation/prepare.py prepare --dataset_config $(CONFIG)

status:  ## show which dataset sources are missing: make status config/datasets/<name>.yaml
	$(require_config)
	uv run python data_preparation/prepare.py status --dataset_config $(CONFIG)

training:  ## train: make training config/<run>.yaml
	$(require_config)
	uv run python training/train.py --config $(CONFIG)

evaluate:  ## samples (and benchmarks with EVAL_TASKS=a,b) for a checkpoint: make evaluate <checkpoint.pth>
	$(require_checkpoint)
	uv run python evaluation/evaluate.py --checkpoint $(CHECKPOINT) --tasks "$(EVAL_TASKS)"

ifneq ($(CONFIG)$(CHECKPOINT),)
.PHONY: $(CONFIG) $(CHECKPOINT)
$(CONFIG) $(CHECKPOINT):  # the path argument is a goal too; nothing to do for it
	@:
endif
