# DeepVQE-GGML Makefile
# =====================
#
# GGML inference targets:
#   build-ggml     Build C++ inference binary
#   build-shared   Build shared library (libdeepvqe.so)
#   test-ggml      Run GGML block tests against PyTorch intermediates
#
# Training (delegated to train/Makefile):
#   build-docker   Build Docker training image
#   train          Train on DNS5 minimal subset
#   export         Export checkpoint to GGUF
#   train-<target> Any train/Makefile target (e.g. make train-test)

# ── GGML Build ──────────────────────────────────────────────────────────────

.PHONY: build-ggml
build-ggml: ## Build GGML C++ inference binary
	nix develop -c bash -c 'cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$$(nproc)'

.PHONY: build-shared
build-shared: ## Build shared library (libdeepvqe.so)
	nix develop -c bash -c 'cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release -DDEEPVQE_BUILD_SHARED=ON && cmake --build build -j$$(nproc)'

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

.PHONY: test-quantize
test-quantize: build-ggml ## Compare F32 vs Q8_0 quantized model outputs
	ggml/build/test_quantize \
		--f32 deepvqe.gguf --q8 deepvqe_q8.gguf \
		--input-npy intermediates/pytorch/mic_stft.npy intermediates/pytorch/ref_stft.npy

.PHONY: test-streaming
test-streaming: build-ggml ## PCM batch-vs-streaming equivalence test
	ggml/build/test_streaming deepvqe.gguf \
		--audio-dirs datasets_fullband/clean datasets_fullband/noise

NPROC_MINUS_1 := $(shell echo $$(($$(nproc) - 1)))
GGML_ENH_DIR  := ggml_eval

.PHONY: eval-ggml
eval-ggml: build-ggml ## Run GGML model on val data, score with MOS models
	@mkdir -p $(GGML_ENH_DIR)
	GGML_NTHREADS=$(NPROC_MINUS_1) ggml/build/eval_graph deepvqe.gguf \
		--val-dir eval_output/val_audio \
		--save-dir $(GGML_ENH_DIR)
	$(MAKE) -C train eval-ggml GGML_ENH_DIR=../$(GGML_ENH_DIR)

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
