# DeepVQE-GGML Makefile
# =====================
#
# Targets:
#   build          Build the Docker training image
#   test           Quick smoke test (dummy data, 2 epochs)
#   train-minimal  Train on DNS5 minimal subset
#   train-full     Train on full dataset (mount custom data dir)
#   eval           Evaluate a checkpoint
#   export         Export checkpoint to GGUF
#   download-data  Download DNS5 minimal dataset
#   test-model     Run model unit tests
#   test-data      Run data pipeline tests
#   tensorboard    Launch TensorBoard viewer
#   shell          Open a shell in the container
#   clean          Remove checkpoints, logs, eval output

IMAGE        := deepvqe
CONTAINER    := deepvqe-train
GPU          := all
CONFIG       := configs/default.yaml

# Host directories to mount
DATA_DIR     := $(CURDIR)/datasets_fullband
CKPT_DIR     := $(CURDIR)/checkpoints
LOG_DIR      := $(CURDIR)/logs
EVAL_DIR     := $(CURDIR)/eval_output
CACHE_DIR    := $(CURDIR)/.cache/torch_inductor

# Training overrides
EPOCHS       ?=
BATCH_SIZE   ?=
EXTRA_ARGS   ?=

DOCKER_RUN := docker run --rm --device nvidia.com/gpu=$(GPU) \
	--name $(CONTAINER) \
	--shm-size=4g \
	--device /dev/fuse --cap-add SYS_ADMIN \
	-e TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
	-e TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor \
	-v $(CURDIR):/workspace/deepvqe \
	-v $(DATA_DIR):/workspace/deepvqe/datasets_fullband \
	-v $(CKPT_DIR):/workspace/deepvqe/checkpoints \
	-v $(LOG_DIR):/workspace/deepvqe/logs \
	-v $(EVAL_DIR):/workspace/deepvqe/eval_output \
	-v $(CACHE_DIR):/cache/torch_inductor

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build Docker training image
	docker build -t $(IMAGE) .

# ── Training ─────────────────────────────────────────────────────────────────

.PHONY: test
test: build ## Smoke test: dummy data, 2 epochs, batch 4
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) --dummy $(EXTRA_ARGS)

.PHONY: train-minimal
train-minimal: build ## Train on DNS5 minimal subset
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) $(EXTRA_ARGS)

.PHONY: train-full
train-full: build ## Train on full dataset (set DATA_DIR= to override)
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) $(EXTRA_ARGS)

.PHONY: overfit
overfit: build ## Overfit test: 8 real audio examples with varied delays
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config configs/overfit.yaml --overfit-real $(EXTRA_ARGS)

# ── Evaluation & Export ──────────────────────────────────────────────────────

CHECKPOINT ?= checkpoints/best.pt

.PHONY: eval
eval: build ## Evaluate a checkpoint (set CHECKPOINT=path)
	$(DOCKER_RUN) $(IMAGE) \
		python eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(EXTRA_ARGS)

.PHONY: listen
listen: build ## Generate WAV audio from checkpoint (set CHECKPOINT=path)
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/listen.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(EXTRA_ARGS)

.PHONY: export
export: build ## Export checkpoint to GGUF (set CHECKPOINT=path)
	$(DOCKER_RUN) \
		-v $(CURDIR):/workspace/deepvqe/output \
		$(IMAGE) \
		python export_ggml.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
			--output output/deepvqe.gguf $(EXTRA_ARGS)

# ── GGML Build ──────────────────────────────────────────────────────────────

.PHONY: build-ggml
build-ggml: ## Build GGML C++ inference binary
	nix develop -c bash -c 'cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$$(nproc)'

.PHONY: test-ggml
test-ggml: build-ggml ## Run all GGML block tests against PyTorch intermediates
	@echo "=== FE ==="
	ggml/build/test_fe --input intermediates/blocks/fe_mic_input.npy --expected intermediates/blocks/fe_mic_output.npy
	@echo "=== Encoder ==="
	ggml/build/test_encoder --gguf deepvqe.gguf --block mic_enc1 --input intermediates/blocks/mic_enc1_input.npy --expected intermediates/blocks/mic_enc1_output.npy
	@echo "=== Bottleneck ==="
	ggml/build/test_bottleneck --gguf deepvqe.gguf --input intermediates/blocks/bottleneck_input.npy --expected intermediates/blocks/bottleneck_output.npy
	@echo "=== Decoder ==="
	ggml/build/test_decoder --gguf deepvqe.gguf --block dec5 --input-0 intermediates/blocks/dec5_input_0.npy --input-1 intermediates/blocks/dec5_input_1.npy --expected intermediates/blocks/dec5_output.npy
	@echo "=== CCM ==="
	ggml/build/test_ccm --input-mask intermediates/blocks/ccm_input_0.npy --input-stft intermediates/blocks/ccm_input_1.npy --expected intermediates/blocks/ccm_output.npy
	@echo "=== AlignBlock ==="
	ggml/build/test_align --gguf deepvqe.gguf --input-mic intermediates/blocks/align_input_0.npy --input-ref intermediates/blocks/align_input_1.npy --expected intermediates/blocks/align_output.npy
	@echo "=== All block tests passed ==="

# ── GGML Comparison ─────────────────────────────────────────────────────────

BLOCK ?= mic_enc1

.PHONY: compare-pt
compare-pt: build ## Export PyTorch intermediates for GGML comparison
	$(DOCKER_RUN) $(IMAGE) \
		python ggml/compare.py --mode pytorch --checkpoint $(CHECKPOINT) \
			--output intermediates/pytorch --use-audio $(EXTRA_ARGS)

.PHONY: compare-block
compare-block: build ## Export single block I/O (set BLOCK=mic_enc1)
	$(DOCKER_RUN) $(IMAGE) \
		python ggml/compare.py --mode block --checkpoint $(CHECKPOINT) \
			--block $(BLOCK) --output intermediates/blocks --use-audio $(EXTRA_ARGS)

# ── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test-model
test-model: build ## Run model unit tests
	$(DOCKER_RUN) $(IMAGE) python test_model.py

.PHONY: test-data
test-data: build ## Run data pipeline tests
	$(DOCKER_RUN) $(IMAGE) python test_data.py

.PHONY: test-blocks
test-blocks: build ## Run block-level verification tests
	$(DOCKER_RUN) $(IMAGE) python test_blocks.py

.PHONY: test-ccm
test-ccm: build ## Run CCM learning isolation tests
	$(DOCKER_RUN) $(IMAGE) python test_ccm_learning.py

.PHONY: tests
tests: test-model test-data test-blocks test-ccm ## Run all unit tests

# ── Data ─────────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: ## Download DNS5 minimal dataset
	bash scripts/download_dns5_minimal.sh $(DATA_DIR)

# ── Run arbitrary commands ──────────────────────────────────────────────────

CMD ?= python -c 'import torch; print(torch.cuda.is_available())'

.PHONY: run
run: build ## Run arbitrary command: make run CMD="python script.py"
	./scripts/docker-run.sh $(CMD)

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: test-erle
test-erle: build ## Test ERLE and delay estimation with real speech
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/test_erle.py $(EXTRA_ARGS)

.PHONY: check
check: build ## Check training progress (loss, entropy, grad norms)
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/check_training.py $(EXTRA_ARGS)

.PHONY: report
report: build ## Training report (summary, scalars, loss, gradients, export)
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/report_training.py $(EXTRA_ARGS)

.PHONY: diagnose
diagnose: build ## Diagnose decoder activations from overfit checkpoint
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/diagnose_decoder.py $(EXTRA_ARGS)

.PHONY: tensorboard
tensorboard: ## Launch TensorBoard on port 6006
	docker run --rm -p 6006:6006 \
		-v $(LOG_DIR):/logs \
		tensorflow/tensorflow \
		tensorboard --logdir /logs --bind_all

.PHONY: shell
shell: build ## Open interactive shell in the container
	./scripts/docker-run.sh --docker-args "-it" bash

.PHONY: clean
clean: ## Remove checkpoints, logs, and eval output
	rm -rf checkpoints/ logs/ eval_output/

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
