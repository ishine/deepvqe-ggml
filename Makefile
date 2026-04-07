# DeepVQE-GGML Makefile
# =====================
#
# Main targets:
#   eval           Full reproducible pipeline: export → build → test → score
#   build-ggml     Build C++ with CUDA (in Docker)
#
# Training (delegated to train/Makefile):
#   build-docker   Build Docker training image
#   train          Train on DNS5 minimal subset
#   train-<target> Any train/Makefile target

# ── Configuration ──────────────────────────────────────────────────────────

N_EVAL     ?= 1000
GGML_MODEL ?= deepvqe.gguf

# ── GGML Build ──────────────────────────────────────────────────────────────

.PHONY: build-ggml
build-ggml: ## Build GGML C++ with CUDA (in Docker)
	$(MAKE) -C train build-ggml

.PHONY: build-ggml-cpu
build-ggml-cpu: ## Build GGML C++ for CPU (host, via nix)
	nix develop -c bash -c 'cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$$(nproc)'

.PHONY: build-shared
build-shared: ## Build shared library (libdeepvqe.so)
	nix develop -c bash -c 'cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release -DDEEPVQE_BUILD_SHARED=ON && cmake --build build -j$$(nproc)'

# ── Evaluation Pipeline ────────────────────────────────────────────────────
#
# Full pipeline from checkpoint + audio samples:
#   1. Export checkpoint → deepvqe.gguf
#   2. Build GGML C++ with CUDA
#   3. Generate val_audio + PyTorch enhanced + PyTorch scores
#   4. Generate block-level PyTorch intermediates
#   5. Run GGML validation tests (STFT, block tests, streaming)
#   6. Run GGML inference on val_audio
#   7. Score GGML enhanced + compare with PyTorch

.PHONY: eval
eval: eval-export eval-pt eval-test eval-ggml ## Full pipeline: export → test → score

.PHONY: eval-export
eval-export: build-ggml ## Export checkpoint to GGUF + build GGML
	$(MAKE) -C train export

.PHONY: eval-pt
eval-pt: ## PyTorch: generate val_audio + pt_enhanced + scores
	$(MAKE) -C train eval-pt EXTRA_ARGS="--max-samples $(N_EVAL)"

.PHONY: eval-test
eval-test: build-ggml ## GGML validation: block tests + STFT + streaming
	$(MAKE) -C train gen-test-data
	$(MAKE) -C train test-ggml
	$(MAKE) -C train test-stft
	$(MAKE) -C train test-streaming

.PHONY: eval-ggml
eval-ggml: build-ggml ## GGML inference + score + compare with PyTorch
	$(MAKE) -C train eval-ggml-infer N_EVAL=$(N_EVAL)
	$(MAKE) -C train eval-score \
		ENH_DIR=../eval_output/ggml_enhanced \
		SCORE_OUTPUT=../eval_output/scores/ggml_scores.json \
		SCORE_LABEL=ggml_f32 \
		EXTRA_ARGS="--compare ../eval_output/scores/pt_scores.json --max-samples $(N_EVAL)"

.PHONY: eval-clean
eval-clean: ## Remove all eval output and stale directories
	rm -rf ggml_eval ggml_eval_f32 ggml_eval_q8 eval_output/py_enhanced

# ── Training delegation ─────────────────────────────────────────────────────

.PHONY: build-docker
build-docker: ## Build Docker training image
	$(MAKE) -C train build

.PHONY: train
train: ## Train on DNS5 minimal subset (delegates to train/)
	$(MAKE) -C train train-minimal

.PHONY: export
export: ## Export checkpoint to GGUF (delegates to train/)
	$(MAKE) -C train export

.PHONY: compare-pt
compare-pt: ## Export PyTorch intermediates for GGML comparison
	$(MAKE) -C train compare-pt

.PHONY: compare-block
compare-block: ## Export single block I/O for GGML comparison
	$(MAKE) -C train compare-block

# Catch-all: make train-<target> delegates to make -C train <target>
train-%:
	$(MAKE) -C train $*

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
