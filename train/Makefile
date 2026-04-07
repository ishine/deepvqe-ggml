# DeepVQE Training Makefile
# ========================
#
# All targets run inside Docker (NGC PyTorch + CUDA).
#
# Targets:
#   build          Build the Docker training image
#   test           Quick smoke test (dummy data, 2 epochs)
#   train          Train on DNS5 dataset
#   overfit        Overfit test (8 real audio examples)
#   eval           Evaluate a checkpoint
#   listen         Generate WAV audio
#   export         Export checkpoint to GGUF
#   test-model     Run model unit tests
#   test-data      Run data pipeline tests
#   test-blocks    Run block-level verification tests
#   test-ccm       Run CCM learning isolation tests
#   tests          Run all unit tests
#   download-data  Download DNS5 16kHz from HuggingFace
#   score-dnsmos   Score clean speech with DNSMOS P.835
#   tensorboard    Launch TensorBoard viewer
#   shell          Open a shell in the container
#   clean          Remove checkpoints, logs, eval output

IMAGE        := deepvqe
IMAGE_ONNX   := deepvqe-onnx
GPU          := all
CONFIG       := configs/default.yaml

# Host directories (relative to project root)
PROJECT_ROOT := $(CURDIR)/..
DATA_DIR     := $(PROJECT_ROOT)/datasets_fullband
CKPT_DIR     := $(PROJECT_ROOT)/checkpoints
LOG_DIR      := $(PROJECT_ROOT)/logs
EVAL_DIR     := $(PROJECT_ROOT)/eval_output
CACHE_DIR    := $(PROJECT_ROOT)/.cache/torch_inductor

# Training overrides
EPOCHS       ?=
BATCH_SIZE   ?=
EXTRA_ARGS   ?=

# Project root is mounted at /workspace/deepvqe.
# -w sets working directory to train/ so Python imports resolve correctly.
# DOCKER_TRAIN is for long-running training jobs (uses deepvqe-train name).
# DOCKER_UTIL is for short-lived utility tasks (export, eval, tests) so they
# don't conflict with a running training container.
define DOCKER_BASE
docker run --rm --device nvidia.com/gpu=$(GPU) \
	--shm-size=4g \
	-e TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
	-e TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor \
	-v $(PROJECT_ROOT):/workspace/deepvqe \
	-v $(DATA_DIR):/workspace/deepvqe/datasets_fullband \
	-v $(CKPT_DIR):/workspace/deepvqe/checkpoints \
	-v $(LOG_DIR):/workspace/deepvqe/logs \
	-v $(EVAL_DIR):/workspace/deepvqe/eval_output \
	-v $(CACHE_DIR):/cache/torch_inductor \
	-w /workspace/deepvqe/train
endef

DOCKER_TRAIN = $(DOCKER_BASE) --name deepvqe-train
DOCKER_UTIL  = $(DOCKER_BASE) --name deepvqe-util

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build Docker training image
	docker build -t $(IMAGE) -f Dockerfile $(PROJECT_ROOT)

.PHONY: build-onnx
build-onnx: ## Build ONNX scoring image (CUDA 12)
	docker build -t $(IMAGE_ONNX) -f Dockerfile.onnx $(PROJECT_ROOT)

# ── Training ─────────────────────────────────────────────────────────────────

.PHONY: test
test: build ## Smoke test: dummy data, 2 epochs, batch 4
	$(DOCKER_TRAIN) $(IMAGE) \
		python train.py --config $(CONFIG) --dummy $(EXTRA_ARGS)

.PHONY: train
train: build ## Train on DNS5 dataset
	$(DOCKER_TRAIN) $(IMAGE) \
		python train.py --config $(CONFIG) $(EXTRA_ARGS)

.PHONY: overfit
overfit: build ## Overfit test: 8 real audio examples with varied delays
	$(DOCKER_TRAIN) $(IMAGE) \
		python train.py --config configs/overfit.yaml --overfit-real $(EXTRA_ARGS)

# ── Evaluation & Export ──────────────────────────────────────────────────────

CHECKPOINT ?= checkpoints/best.pt

.PHONY: eval
eval: build ## Evaluate a checkpoint (set CHECKPOINT=path)
	$(DOCKER_UTIL) $(IMAGE) \
		python eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(EXTRA_ARGS)

.PHONY: listen
listen: build ## Generate WAV audio from checkpoint (set CHECKPOINT=path)
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/listen.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(EXTRA_ARGS)

.PHONY: eval-pt
eval-pt: build ## PyTorch eval: save val_audio + pt_enhanced + pt_scores.json
	$(DOCKER_UTIL) $(IMAGE) \
		python eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
			--save-val-audio $(EXTRA_ARGS)

.PHONY: build-ggml
build-ggml: build ## Build GGML C++ with CUDA inside Docker
	$(DOCKER_UTIL) $(IMAGE) bash -c '\
		cd /workspace/deepvqe/ggml && \
		cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DDEEPVQE_CUDA=ON && \
		cmake --build build-cuda -j$$(nproc)'

GGML_MODEL ?= deepvqe.gguf
N_EVAL     ?= 1000

.PHONY: eval-ggml-infer
eval-ggml-infer: build-ggml ## Run GGML CUDA inference on val data
	$(DOCKER_UTIL) $(IMAGE) bash -c '\
		mkdir -p /workspace/deepvqe/eval_output/ggml_enhanced && \
		/workspace/deepvqe/ggml/build-cuda/eval_graph \
			/workspace/deepvqe/$(GGML_MODEL) \
			--val-dir /workspace/deepvqe/eval_output/val_audio \
			--save-dir /workspace/deepvqe/eval_output/ggml_enhanced \
			--max $(N_EVAL)'

SCORE_LABEL  ?= unknown
SCORE_OUTPUT ?= ../eval_output/scores/scores.json
ENH_DIR      ?= ../eval_output/ggml_enhanced

.PHONY: eval-score
eval-score: build ## Score enhanced audio (ERLE/PESQ/STOI/DNSMOS/AECMOS)
	$(DOCKER_UTIL) $(IMAGE) \
		python3 scripts/eval_score.py \
			--val-dir ../eval_output/val_audio \
			--enh-dir $(ENH_DIR) \
			--label $(SCORE_LABEL) \
			--output $(SCORE_OUTPUT) $(EXTRA_ARGS)

# ── GGML Validation Tests (run inside Docker with CUDA) ───────────────────

TEST_BLOCKS := fe_mic mic_enc1 bottleneck dec5 ccm align

.PHONY: gen-test-data
gen-test-data: build ## Generate PyTorch reference data for GGML tests
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/compare.py --mode pytorch --checkpoint $(CHECKPOINT) \
			--output ../intermediates/pytorch --use-audio
	@for block in $(TEST_BLOCKS); do \
		echo "Exporting block: $$block"; \
		$(DOCKER_UTIL) $(IMAGE) \
			python scripts/compare.py --mode block --checkpoint $(CHECKPOINT) \
				--block $$block --output ../intermediates/blocks --use-audio; \
	done
	$(DOCKER_UTIL) $(IMAGE) \
		python3 scripts/gen_stft_ref.py \
			--output-dir ../ggml/test_data \
			--checkpoint $(CHECKPOINT) \
			--val-dir ../eval_output/val_audio \
			--n-samples 5

.PHONY: test-ggml
test-ggml: build-ggml ## Run GGML block tests against PyTorch intermediates
	$(DOCKER_UTIL) $(IMAGE) bash -c '\
		cd /workspace/deepvqe && \
		echo "=== FE ===" && \
		ggml/build-cuda/test_fe --input intermediates/blocks/fe_mic_input.npy --expected intermediates/blocks/fe_mic_output.npy && \
		echo "=== Encoder ===" && \
		ggml/build-cuda/test_encoder --gguf deepvqe.gguf --block mic_enc1 --input intermediates/blocks/mic_enc1_input.npy --expected intermediates/blocks/mic_enc1_output.npy && \
		echo "=== Bottleneck ===" && \
		ggml/build-cuda/test_bottleneck --gguf deepvqe.gguf --input intermediates/blocks/bottleneck_input.npy --expected intermediates/blocks/bottleneck_output.npy && \
		echo "=== Decoder ===" && \
		ggml/build-cuda/test_decoder --gguf deepvqe.gguf --block dec5 --input-0 intermediates/blocks/dec5_input_0.npy --input-1 intermediates/blocks/dec5_input_1.npy --expected intermediates/blocks/dec5_output.npy && \
		echo "=== CCM ===" && \
		ggml/build-cuda/test_ccm --input-mask intermediates/blocks/ccm_input_0.npy --input-stft intermediates/blocks/ccm_input_1.npy --expected intermediates/blocks/ccm_output.npy && \
		echo "=== AlignBlock ===" && \
		ggml/build-cuda/test_align --gguf deepvqe.gguf --input-mic intermediates/blocks/align_input_0.npy --input-ref intermediates/blocks/align_input_1.npy --expected intermediates/blocks/align_output.npy && \
		echo "=== All block tests passed ==="'

.PHONY: test-stft
test-stft: build-ggml ## Run STFT + end-to-end model comparison tests
	$(DOCKER_UTIL) $(IMAGE) bash -c '\
		cd /workspace/deepvqe && \
		ggml/build-cuda/test_stft \
			--ref-input ggml/test_data/stft_input.npy \
			--ref-stft ggml/test_data/stft_ref.npy \
			--model deepvqe.gguf \
			--val-dir eval_output/val_audio \
			--py-ref-dir ggml/test_data \
			--n-samples 5'

.PHONY: test-streaming
test-streaming: build-ggml ## PCM batch-vs-streaming equivalence test
	$(DOCKER_UTIL) $(IMAGE) bash -c '\
		cd /workspace/deepvqe && \
		ggml/build-cuda/test_streaming deepvqe.gguf \
			--audio-dirs datasets_fullband/clean datasets_fullband/noise'

.PHONY: export
export: build ## Export checkpoint to GGUF (set CHECKPOINT=path)
	$(DOCKER_UTIL) $(IMAGE) \
		python export_ggml.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
			--output ../deepvqe.gguf $(EXTRA_ARGS)

# ── GGML Comparison ─────────────────────────────────────────────────────────

BLOCK ?= mic_enc1

.PHONY: compare-pt
compare-pt: build ## Export PyTorch intermediates for GGML comparison
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/compare.py --mode pytorch --checkpoint $(CHECKPOINT) \
			--output ../intermediates/pytorch --use-audio $(EXTRA_ARGS)

.PHONY: compare-block
compare-block: build ## Export single block I/O (set BLOCK=mic_enc1)
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/compare.py --mode block --checkpoint $(CHECKPOINT) \
			--block $(BLOCK) --output ../intermediates/blocks --use-audio $(EXTRA_ARGS)

# ── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test-model
test-model: build ## Run model unit tests
	$(DOCKER_UTIL) $(IMAGE) python test_model.py

.PHONY: test-data
test-data: build ## Run data pipeline tests
	$(DOCKER_UTIL) $(IMAGE) python test_data.py

.PHONY: test-blocks
test-blocks: build ## Run block-level verification tests
	$(DOCKER_UTIL) $(IMAGE) python test_blocks.py

.PHONY: test-ccm
test-ccm: build ## Run CCM learning isolation tests
	$(DOCKER_UTIL) $(IMAGE) python test_ccm_learning.py

.PHONY: test-optim
test-optim: build ## Run optimizer verification tests
	$(DOCKER_UTIL) $(IMAGE) python test_optimizer.py

.PHONY: tests
tests: test-model test-data test-blocks test-ccm test-optim ## Run all unit tests

# ── Data ─────────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: build ## Download DNS5 16kHz dataset from HuggingFace
	$(DOCKER_UTIL) $(IMAGE) python scripts/download_dns5_hf.py $(EXTRA_ARGS)

.PHONY: score-dnsmos
score-dnsmos: build-onnx ## Score clean files with DNSMOS P.835 (GPU)
	$(DOCKER_UTIL) $(IMAGE_ONNX) python3 scripts/score_dnsmos.py $(EXTRA_ARGS)

# ── Run arbitrary commands ──────────────────────────────────────────────────

CMD ?= python -c 'import torch; print(torch.cuda.is_available())'

.PHONY: run
run: build ## Run arbitrary command: make run CMD="python script.py"
	./scripts/docker-run.sh $(CMD)

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: test-erle
test-erle: build ## Test ERLE and delay estimation with real speech
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/test_erle.py $(EXTRA_ARGS)

.PHONY: check
check: build ## Check training progress (loss, entropy, grad norms)
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/check_training.py $(EXTRA_ARGS)

.PHONY: report
report: build ## Training report (summary, scalars, loss, gradients, export)
	$(DOCKER_UTIL) $(IMAGE) \
		python scripts/report_training.py $(EXTRA_ARGS)

.PHONY: diagnose
diagnose: build ## Diagnose decoder activations from overfit checkpoint
	$(DOCKER_UTIL) $(IMAGE) \
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
	rm -rf $(CKPT_DIR) $(LOG_DIR) $(EVAL_DIR)

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
