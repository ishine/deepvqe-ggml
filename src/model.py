import torch
import torch.nn as nn

from src.align import AlignBlock
from src.blocks import (
    FE,
    Bottleneck,
    DecoderBlock,
    EncoderBlock,
)
from src.ccm import CCM


class DeepVQEAEC(nn.Module):
    """DeepVQE with far-end branch and soft delay alignment for AEC.

    Architecture:
        Mic encoder: 5 blocks (2→64→128→128→128→128)
        Far-end encoder: 2 blocks (2→32→128)
        AlignBlock after enc2: cross-attention soft delay
        Enc3 input: 256 channels (concat mic_enc2 + aligned_far)
        Bottleneck: GRU(1152→576) + Linear(576→1152)
        Decoder: 5 blocks with sub-pixel conv (128→128→128→64→64)
        Mask head: 1x1 Conv2d(64→27), no BN (preserves spatial contrast)
        CCM: 27ch → 3×3 complex convolving mask
    """

    def __init__(
        self,
        mic_channels=None,
        far_channels=None,
        align_hidden=32,
        dmax=32,
        power_law_c=0.3,
    ):
        super().__init__()
        if mic_channels is None:
            mic_channels = [2, 64, 128, 128, 128, 128]
        if far_channels is None:
            far_channels = [2, 32, 128]

        # Feature extraction
        self.fe_mic = FE(c=power_law_c)
        self.fe_ref = FE(c=power_law_c)

        # Mic encoder blocks 1-2
        self.mic_enc1 = EncoderBlock(mic_channels[0], mic_channels[1])
        self.mic_enc2 = EncoderBlock(mic_channels[1], mic_channels[2])

        # Far-end encoder blocks 1-2
        self.far_enc1 = EncoderBlock(far_channels[0], far_channels[1])
        self.far_enc2 = EncoderBlock(far_channels[1], far_channels[2])

        # Alignment
        self.align = AlignBlock(
            in_channels=mic_channels[2],  # 128
            hidden_channels=align_hidden,
            dmax=dmax,
        )

        # Mic encoder blocks 3-5 (block 3 takes concat: 128+128=256)
        self.mic_enc3 = EncoderBlock(mic_channels[2] * 2, mic_channels[3])  # 256→128
        self.mic_enc4 = EncoderBlock(mic_channels[3], mic_channels[4])
        self.mic_enc5 = EncoderBlock(mic_channels[4], mic_channels[5])

        # Bottleneck: 128 channels * 9 freq bins = 1152
        self.bottleneck = Bottleneck(mic_channels[5] * 9, mic_channels[5] * 9 // 2)

        # Decoder blocks (mirror encoder)
        self.dec5 = DecoderBlock(mic_channels[5], mic_channels[4])  # 128→128
        self.dec4 = DecoderBlock(mic_channels[4], mic_channels[3])  # 128→128
        self.dec3 = DecoderBlock(mic_channels[3], mic_channels[2])  # 128→128
        self.dec2 = DecoderBlock(mic_channels[2], mic_channels[1])  # 128→64
        self.dec1 = DecoderBlock(mic_channels[1], mic_channels[1])  # 64→64

        # Mask estimation head: BN-free 1x1 conv to preserve spatial contrast.
        # Decoder BN normalizes per-channel across T-F positions, smoothing out
        # frequency-dependent variation. The mask head bypasses this — it maps
        # 64 BN-normalized feature channels to 27 CCM mask channels without
        # normalization, allowing it to produce spatially varying masks.
        self.mask_head = nn.Conv2d(mic_channels[1], 27, 1)

        # Complex Convolving Mask
        self.ccm = CCM()

        self._init_ccm_identity()

    def _init_ccm_identity(self):
        """Initialize mask_head bias so the CCM mask starts as identity (passthrough).

        The 27-ch mask is reshaped as (3 basis, 9 kernel).  Basis vectors
        v_real=[1,-0.5,-0.5] and v_imag=[0,√3/2,-√3/2] sum to zero, so
        default init (similar values across the 3 groups) produces near-zero
        mask magnitude.  Fix: set the current-frame kernel element of the
        first basis (r=0, v_real=1, v_imag=0) to 1, giving H_real[center]=1.

        CCM uses causal ZeroPad2d([1,1,2,0]) with 3×3 kernel:
          m=0: t-2, m=1: t-1, m=2: t (current frame)
          n=0: f-1, n=1: f (current freq), n=2: f+1
        So current (t, f) = kernel index 2*3+1 = 7, NOT 4.

        mask_head is a 1x1 Conv2d(64, 27) — bias alone sets the identity.
        """
        with torch.no_grad():
            # Small random weights (default init) allow gradient flow.
            # Bias sets the identity mask; weights add feature-dependent corrections.
            self.mask_head.bias.zero_()
            self.mask_head.bias[7] = 1.0   # r=0, current (t,f)

    def forward(self, mic_stft, ref_stft, return_delay=False):
        """
        mic_stft: (B, 257, T, 2) — microphone STFT
        ref_stft: (B, 257, T, 2) — far-end reference STFT

        Returns:
            enhanced: (B, 257, T, 2) — enhanced STFT
            delay_dist: (B, T, dmax) — delay distribution (if return_delay=True)
        """
        # Feature extraction
        mic_fe = self.fe_mic(mic_stft)  # (B, 2, T, 257)
        ref_fe = self.fe_ref(ref_stft)  # (B, 2, T, 257)

        # Mic encoder 1-2
        mic_e1 = self.mic_enc1(mic_fe)  # (B, 64, T, 129)
        mic_e2 = self.mic_enc2(mic_e1)  # (B, 128, T, 65)

        # Far-end encoder 1-2
        far_e1 = self.far_enc1(ref_fe)  # (B, 32, T, 129)
        far_e2 = self.far_enc2(far_e1)  # (B, 128, T, 65)

        # Alignment
        align_result = self.align(mic_e2, far_e2, return_delay=return_delay)
        if return_delay:
            aligned_far, delay_dist = align_result
        else:
            aligned_far = align_result

        # Concat mic + aligned far-end
        concat = torch.cat([mic_e2, aligned_far], dim=1)  # (B, 256, T, 65)

        # Mic encoder 3-5
        mic_e3 = self.mic_enc3(concat)  # (B, 128, T, 33)
        mic_e4 = self.mic_enc4(mic_e3)  # (B, 128, T, 17)
        mic_e5 = self.mic_enc5(mic_e4)  # (B, 128, T, 9)

        # Bottleneck
        bn = self.bottleneck(mic_e5)  # (B, 128, T, 9)

        # Decoder with skip connections (trim freq to match encoder)
        d5 = self.dec5(bn, mic_e5)[..., : mic_e4.shape[-1]]
        d4 = self.dec4(d5, mic_e4)[..., : mic_e3.shape[-1]]
        d3 = self.dec3(d4, mic_e3)[..., : mic_e2.shape[-1]]
        d2 = self.dec2(d3, mic_e2)[..., : mic_e1.shape[-1]]
        d1_feat = self.dec1(d2, mic_e1)[..., : mic_fe.shape[-1]]

        # Mask head: BN-free projection to 27 CCM channels
        d1 = self.mask_head(d1_feat)

        # Apply complex convolving mask
        enhanced = self.ccm(d1, mic_stft)  # (B, 257, T, 2)

        if return_delay:
            return enhanced, delay_dist, d1
        return enhanced

    @classmethod
    def from_config(cls, cfg):
        """Create model from a Config object."""
        return cls(
            mic_channels=cfg.model.mic_channels,
            far_channels=cfg.model.far_channels,
            align_hidden=cfg.model.align_hidden,
            dmax=cfg.model.dmax,
            power_law_c=cfg.model.power_law_c,
        )
