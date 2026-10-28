.PHONY: install lint test download pull repro exp-show metrics plots dag push

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
