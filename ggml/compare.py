"""Layer-by-layer comparison between PyTorch and GGML inference.

Registers forward hooks on all PyTorch layers to capture intermediate
activations, then compares against GGML binary dumps.

Usage:
    # 1a. Generate PyTorch intermediates (random input)
    python ggml/compare.py --mode pytorch --checkpoint best.pt --output intermediates/

    # 1b. Generate PyTorch intermediates (real audio input)
    python ggml/compare.py --mode pytorch --checkpoint best.pt --use-audio --output intermediates/

    # 1c. Export single block input/output for isolated C++ verification
    python ggml/compare.py --mode block --checkpoint best.pt --block mic_enc1 --output intermediates/blocks/

    # 2. Run GGML inference with dumps
    ./ggml/deepvqe model.gguf --dump-intermediates

    # 3. Compare
    python ggml/compare.py --mode compare --pytorch-dir intermediates/ --ggml-dir ggml_intermediates/
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path when running from ggml/ subdir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model import DeepVQEAEC
from src.stft import istft
from train import load_checkpoint


def capture_intermediates(model, mic_stft, ref_stft):
    """Run forward pass and capture all intermediate activations.

    Returns:
        output: model output tensor
        intermediates: OrderedDict of {layer_name: tensor}
    """
    intermediates = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Store first element for modules that return tuples
                intermediates[name] = output[0].detach().cpu()
            else:
                intermediates[name] = output.detach().cpu()
        return hook_fn

    # Register hooks on all named modules
    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        output = model(mic_stft, ref_stft)

    # Remove hooks
    for h in hooks:
        h.remove()

    return output, intermediates


def capture_block_io(model, mic_stft, ref_stft, block_name):
    """Run forward pass and capture a single block's input(s) and output.

    Returns:
        block_inputs: list of tensors (input(s) to the block)
        block_output: tensor (output of the block)
        model_output: tensor (full model output)
    """
    block_inputs = []
    block_output = None

    def hook_fn(module, input, output):
        nonlocal block_output
        # Capture all input tensors
        for inp in input:
            if isinstance(inp, torch.Tensor):
                block_inputs.append(inp.detach().cpu())
        # Capture output
        if isinstance(output, tuple):
            block_output = output[0].detach().cpu()
        else:
            block_output = output.detach().cpu()

    # Find the target module
    target = None
    for name, module in model.named_modules():
        if name == block_name:
            target = module
            break

    if target is None:
        available = [n for n, _ in model.named_modules() if n]
        raise ValueError(
            f"Block '{block_name}' not found. Available blocks:\n"
            + "\n".join(f"  {n}" for n in available)
        )

    handle = target.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model_output = model(mic_stft, ref_stft)
    handle.remove()

    return block_inputs, block_output, model_output


def save_intermediates(intermediates, output_dir):
    """Save intermediate activations as .npy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, tensor in intermediates.items():
        safe_name = name.replace(".", "_")
        np.save(output_dir / f"{safe_name}.npy", tensor.numpy())

    print(f"Saved {len(intermediates)} intermediate activations to {output_dir}")


def load_intermediates(dir_path):
    """Load intermediate activations from .npy files."""
    dir_path = Path(dir_path)
    intermediates = {}
    for f in sorted(dir_path.glob("*.npy")):
        name = f.stem
        intermediates[name] = np.load(f)
    return intermediates


def compare_intermediates(pytorch_dir, ggml_dir):
    """Compare PyTorch and GGML intermediate activations."""
    pt_data = load_intermediates(pytorch_dir)
    ggml_data = load_intermediates(ggml_dir)

    print(f"PyTorch layers: {len(pt_data)}")
    print(f"GGML layers: {len(ggml_data)}")
    print()

    # Find matching layers
    matched = 0
    max_errors = []

    for name in pt_data:
        if name not in ggml_data:
            continue

        pt_arr = pt_data[name]
        ggml_arr = ggml_data[name]

        if pt_arr.shape != ggml_arr.shape:
            print(f"  {name}: SHAPE MISMATCH pt={pt_arr.shape} ggml={ggml_arr.shape}")
            continue

        max_err = np.max(np.abs(pt_arr - ggml_arr))
        mean_err = np.mean(np.abs(pt_arr - ggml_arr))
        max_errors.append((name, max_err, mean_err))
        matched += 1

        status = "OK" if max_err < 1e-4 else "WARN" if max_err < 1e-2 else "FAIL"
        print(f"  [{status}] {name}: max={max_err:.2e} mean={mean_err:.2e}")

    print(f"\nMatched {matched} layers")
    if max_errors:
        worst = max(max_errors, key=lambda x: x[1])
        print(f"Worst layer: {worst[0]} (max error: {worst[1]:.2e})")
        overall_max = max(e[1] for e in max_errors)
        print(f"Overall max error: {overall_max:.2e}")
        if overall_max < 1e-4:
            print("PASS: All layers within f32 tolerance (1e-4)")
        elif overall_max < 1e-2:
            print("WARN: Some layers exceed f32 tolerance, acceptable for f16")
        else:
            print("FAIL: Errors exceed acceptable tolerance")


def _make_input(cfg, use_audio, sample_idx, frames):
    """Create model input tensors (either random or from dataset)."""
    if use_audio:
        from data.dataset import AECDataset
        ds = AECDataset(cfg, split="val")
        sample = ds[sample_idx]
        mic_stft = sample["mic_stft"].unsqueeze(0)
        ref_stft = sample["ref_stft"].unsqueeze(0)
        print(f"Using real audio (val sample {sample_idx}), shape: {mic_stft.shape}")
        return mic_stft, ref_stft
    else:
        torch.manual_seed(42)
        mic_stft = torch.randn(1, 257, frames, 2)
        ref_stft = torch.randn(1, 257, frames, 2)
        print(f"Using random input (seed=42), shape: {mic_stft.shape}")
        return mic_stft, ref_stft


def generate_pytorch_intermediates(cfg, checkpoint_path, output_dir,
                                   use_audio=False, sample_idx=0, frames=20):
    """Generate and save PyTorch intermediate activations."""
    device = torch.device("cpu")  # Use CPU for deterministic comparison
    model = DeepVQEAEC.from_config(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    mic_stft, ref_stft = _make_input(cfg, use_audio, sample_idx, frames)

    # Save inputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "input_mic_stft.npy", mic_stft.numpy())
    np.save(out_dir / "input_ref_stft.npy", ref_stft.numpy())

    # Also save input as WAV for listening
    try:
        import soundfile as sf
        sr = cfg.audio.sample_rate
        target_len = int(cfg.training.clip_length_sec * sr)
        mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=target_len)
        sf.write(out_dir / "input_mic.wav", mic_wav[0].numpy(), sr, subtype="PCM_16")
    except Exception:
        pass  # WAV export is optional

    output, intermediates = capture_intermediates(model, mic_stft, ref_stft)

    # Save output
    np.save(out_dir / "output.npy", output.detach().cpu().numpy())

    # Save intermediates
    save_intermediates(intermediates, out_dir)

    print(f"Output shape: {output.shape}")
    print(f"Saved inputs, output, and {len(intermediates)} intermediates to {out_dir}")


def generate_block_intermediates(cfg, checkpoint_path, block_name, output_dir,
                                 use_audio=False, sample_idx=0, frames=20):
    """Export a single block's input/output for isolated C++ verification."""
    device = torch.device("cpu")
    model = DeepVQEAEC.from_config(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    mic_stft, ref_stft = _make_input(cfg, use_audio, sample_idx, frames)

    block_inputs, block_output, model_output = capture_block_io(
        model, mic_stft, ref_stft, block_name
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = block_name.replace(".", "_")

    # Save model-level inputs (needed to reproduce the full forward)
    np.save(out_dir / "input_mic_stft.npy", mic_stft.numpy())
    np.save(out_dir / "input_ref_stft.npy", ref_stft.numpy())

    # Save block input(s)
    if len(block_inputs) == 1:
        np.save(out_dir / f"{safe_name}_input.npy", block_inputs[0].numpy())
        print(f"Block input shape: {block_inputs[0].shape}")
    else:
        for j, inp in enumerate(block_inputs):
            np.save(out_dir / f"{safe_name}_input_{j}.npy", inp.numpy())
            print(f"Block input {j} shape: {inp.shape}")

    # Save block output
    np.save(out_dir / f"{safe_name}_output.npy", block_output.numpy())
    print(f"Block output shape: {block_output.shape}")

    print(f"Saved block '{block_name}' I/O to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch vs GGML inference")
    parser.add_argument("--mode", choices=["pytorch", "block", "compare"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint (for pytorch/block modes)")
    parser.add_argument("--output", default="intermediates/pytorch",
                        help="Output dir (for pytorch/block modes)")
    parser.add_argument("--pytorch-dir", default="intermediates/pytorch",
                        help="PyTorch intermediates dir (for compare mode)")
    parser.add_argument("--ggml-dir", default="intermediates/ggml",
                        help="GGML intermediates dir (for compare mode)")
    parser.add_argument("--use-audio", action="store_true",
                        help="Use real audio from AECDataset instead of random noise")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index for --use-audio")
    parser.add_argument("--frames", type=int, default=188,
                        help="Number of STFT frames (default 188 = 3 seconds)")
    parser.add_argument("--block", default=None,
                        help="Block name for block mode (e.g., mic_enc1, align, bottleneck)")
    args = parser.parse_args()

    if args.mode == "pytorch":
        if not args.checkpoint:
            parser.error("--checkpoint required for pytorch mode")
        cfg = load_config(args.config)
        generate_pytorch_intermediates(
            cfg, args.checkpoint, args.output,
            use_audio=args.use_audio, sample_idx=args.sample_idx, frames=args.frames,
        )
    elif args.mode == "block":
        if not args.checkpoint:
            parser.error("--checkpoint required for block mode")
        if not args.block:
            parser.error("--block required for block mode")
        cfg = load_config(args.config)
        generate_block_intermediates(
            cfg, args.checkpoint, args.block, args.output,
            use_audio=args.use_audio, sample_idx=args.sample_idx, frames=args.frames,
        )
    elif args.mode == "compare":
        compare_intermediates(args.pytorch_dir, args.ggml_dir)
