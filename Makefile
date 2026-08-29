# Development entry points. Everything runs through uv (no manual venvs).
.PHONY: setup test typecheck lint

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
