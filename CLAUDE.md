# DeepVQE Project Guidelines

## Project Goal
Train DeepVQE (Indenbom et al., Interspeech 2023) for joint AEC/NS/DR with
a focus on acoustic echo cancellation with soft delay estimation. Deploy via
GGML C++ inference.

## Important Rules
- **Keep README.md up to date** whenever you make structural changes, complete
  checklist items, add new files, or change the architecture/approach.
- Use the real-valued CCM implementation (no `torch.complex`) for GGML compatibility.
- All convolutions must be causal (pad top/left only, no look-ahead).
- Prefer `einops` for tensor reshaping.
- Target 16 kHz, 512 FFT, 256 hop, 257 freq bins.
- Use mixed precision (AMP) for training.
- Log extensively: loss components, gradient norms, delay distributions, audio samples.

## Architecture Quick Reference
- Encoder: 5 mic blocks (2→64→128→128→128→128), 2 far-end (2→32→128)
- AlignBlock: cross-attention soft delay after encoder stage 2
- Encoder block 3 takes 256 input channels (concat of mic + aligned far-end)
- Bottleneck: GRU(1152→576) + Linear(576→1152)
- Decoder: 5 blocks with sub-pixel conv (128,128,128,64,27)
- CCM: 27 channels → 3×3 complex convolving mask (real-valued arithmetic)

## Code Sources
- `deepvqe_xr.py`: Xiaobin-Rong NS-only implementation (clean code base)
- Okrio model.py: AEC reference with far-end branch + AlignBlock

## Known Bugs in Reference Code
- Xiaobin-Rong AlignBlock (line 57): uses `K.shape[1]` (hidden=32) instead of
  `x_ref.shape[1]` (in_channels=128) for weighted sum reshape — must fix.
- Okrio AlignBlock: `torch.zeros()` without `.to(device)` — will fail on GPU.
