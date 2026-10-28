.PHONY: install lint test download pull repro experiments exp-show metrics plots dag push

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

download:  ## refresh data/raw from SMARD + Open-Meteo, then version it
	uv run python -m gridload.download
	uv run dvc add data/raw

pull:  ## fetch data/raw and cached stage outputs from the DVC remote
	uv run dvc pull

repro:  ## rerun only the stages whose inputs changed
	uv run dvc repro

experiments:  ## the three variations from the README, each in a temp copy of the repo
	uv run dvc exp run --temp -n no-weather -S features.weather=false
	uv run dvc exp run --temp -n no-holidays -S features.holidays=false
	uv run dvc exp run --temp -n bigger-trees -S train.num_leaves=127 \
		-S train.min_child_samples=50 -S train.learning_rate=0.03

exp-show:
	uv run dvc exp show --only-changed --md

metrics:  ## compare the workspace with the last commit
	uv run dvc metrics diff --md

plots:  ## writes dvc_plots/index.html
	uv run dvc plots show

dag:
	uv run dvc dag

push:
	uv run dvc push
