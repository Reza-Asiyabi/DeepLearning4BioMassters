.PHONY: install download-data train-all evaluate-all compare test lint clean

# ── Installation ────────────────────────────────────────────────────────────
install:
	conda env create -f environment.yml
	conda run -n biomassters pip install -e .

install-pip:
	pip install -r requirements.txt
	pip install -e .

# ── Data ─────────────────────────────────────────────────────────────────────
download-data:
	python scripts/download_data.py --output-dir data/biomassters --split train
	python scripts/download_data.py --output-dir data/biomassters --split test

download-train:
	python scripts/download_data.py --output-dir data/biomassters --split train

download-test:
	python scripts/download_data.py --output-dir data/biomassters --split test

# ── Training ──────────────────────────────────────────────────────────────────
train-unet:
	python scripts/train.py --config configs/unet.yaml

train-unet3d:
	python scripts/train.py --config configs/unet3d.yaml

train-swin:
	python scripts/train.py --config configs/swin_unet.yaml

train-utae:
	python scripts/train.py --config configs/utae.yaml

train-tempfusion:
	python scripts/train.py --config configs/tempfusionnet.yaml

train-all: train-unet train-unet3d train-swin train-utae train-tempfusion

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate-unet:
	python scripts/evaluate.py \
		--checkpoint results/unet/checkpoints/last.ckpt \
		--config configs/unet.yaml

evaluate-utae:
	python scripts/evaluate.py \
		--checkpoint results/utae/checkpoints/last.ckpt \
		--config configs/utae.yaml

evaluate-all:
	@for model in unet unet3d swin_unet utae tempfusionnet; do \
		echo "Evaluating $$model..."; \
		python scripts/evaluate.py \
			--checkpoint results/$$model/checkpoints/last.ckpt \
			--config configs/$$model.yaml \
			--output-dir results/$$model/ || true; \
	done

# ── Comparison ────────────────────────────────────────────────────────────────
compare:
	python scripts/compare_models.py --results-dir results/ --assets-dir assets/

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=src/biomassters --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

typecheck:
	mypy src/biomassters/ --ignore-missing-imports

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ coverage.xml dist/ build/

clean-results:
	@echo "WARNING: This will delete all trained checkpoints and metrics!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && \
		rm -rf results/*/checkpoints/ || echo "Aborted."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "BioMassters AGB Estimation Benchmark"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Create conda environment and install package"
	@echo "  install-pip      - Install via pip (no conda)"
	@echo "  download-data    - Download train + test splits from HuggingFace"
	@echo "  train-all        - Train all 5 architectures sequentially"
	@echo "  evaluate-all     - Evaluate all trained checkpoints"
	@echo "  compare          - Generate comparison figures and table"
	@echo "  test             - Run unit tests with coverage"
	@echo "  lint             - Check code style with ruff + black"
	@echo "  format           - Auto-format code"
	@echo "  clean            - Remove Python cache files"
