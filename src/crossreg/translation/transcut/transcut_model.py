"""
TransCUT — cross-modality translation with shared Swin-Transformer encoder.

Architecture
------------
::

    Source Image (n₁) + Source ID → Swin Encoder (CLN) → Multi-scale Features
                                                              │
    Target ID ────────────────────────────────────────────────┤
                                                              ▼
                                                   CNN Decoder (AdaIN)
                                                              │
                                                              ▼
                                                    Translated Image (nⱼ_fake)

Training uses PatchNCE (CUT) contrastive loss to preserve structure.

Usage
-----
::

    from crossreg.translation.transcut import TransCUT, TransCUTConfig

    config = TransCUTConfig(num_modalities=6)
    model = TransCUT(config)

    # Training step
    fake = model(source_img, src_id=0, tgt_id=1)
    # NCE features
    feats_q = model.encode(source_img, src_id=0, nce_layers=[0,1,2,3])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from crossreg.models.swin_transformer import SwinTransformer
from crossreg.translation.cut.networks import GANLoss, define_F, init_net
from crossreg.translation.cut.patchnce import PatchNCELoss

from .cln_adain import ModalityIDEmbedding, CLN2d
from .conditional_discriminator import ConditionalPatchDiscriminator
from .decoder import TransCUTDecoder
from .highres_decoder import HighResolutionContentDecoder
from .fullres_residual_decoder import FullResolutionResidualDecoder


# =============================================================================
# Config
# =============================================================================


@dataclass
class TransCUTConfig:
    """Configuration for TransCUT — decoupled from argparse / CUTConfig.

    All parameters have sensible defaults for 256×256 grayscale images
    with up to 6 modalities.
    """

    # --- I/O ---
    input_nc: int = 1
    output_nc: int = 1
    num_modalities: int = 6
    id_embed_dim: int = 64     # modality ID embedding size

    # --- Swin Encoder ---
    img_size: int = 256
    patch_size: int = 4
    embed_dim: int = 96
    depths: tuple = (2, 2, 4, 2)
    num_heads: tuple = (4, 4, 8, 8)
    window_size: tuple = (8, 8)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.3
    ape: bool = False
    spe: bool = False
    rpe: bool = True
    out_indices: tuple = (0, 1, 2, 3)

    # --- Decoder ---
    decoder_style_dim: int = 64
    decoder_variant: str = "legacy"

    # --- Discriminator ---
    ndf: int = 64
    n_layers_D: int = 3
    gan_mode: str = "lsgan"

    # --- NCE ---
    netF_nc: int = 256
    nce_T: float = 0.07
    num_patches: int = 256
    nce_fake_modality: str = "target"

    # --- Loss weights ---
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    lambda_identity: float = 1.0
    lambda_paired: float = 0.0
    lambda_cycle: float = 0.0
    lambda_structure: float = 0.0
    lambda_D_mismatch: float = 0.0
    nce_idt: bool = True

    # --- Training ---
    lr: float = 2e-4
    lr_D: float | None = None
    d_update_freq: int = 2
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"
    n_epochs: int = 200
    n_epochs_decay: int = 200
    epoch_count: int = 1
    lr_decay_iters: int = 50
    gpu_ids: list[int] = field(default_factory=list)


# =============================================================================
# CLN-augmented Swin Encoder wrapper
# =============================================================================


class SwinEncoderWithCLN(nn.Module):
    """Swin-Transformer with CLN injected after patch embedding.

    The CLN receives a modality ID vector and normalises the embedded
    patches so the encoder learns modality-invariant structural features.
    """

    def __init__(self, swin_cfg: TransCUTConfig, mod_embed: ModalityIDEmbedding):
        super().__init__()
        self.swin = SwinTransformer(
            pretrain_img_size=swin_cfg.img_size,
            patch_size=swin_cfg.patch_size,
            in_chans=swin_cfg.input_nc,
            embed_dim=swin_cfg.embed_dim,
            depths=swin_cfg.depths,
            num_heads=swin_cfg.num_heads,
            window_size=swin_cfg.window_size,
            mlp_ratio=swin_cfg.mlp_ratio,
            drop_path_rate=swin_cfg.drop_path_rate,
            ape=swin_cfg.ape,
            spe=swin_cfg.spe,
            rpe=swin_cfg.rpe,
            out_indices=swin_cfg.out_indices,
            patch_norm=False,  # CLN replaces patch_norm
        )
        self.cln = CLN2d(swin_cfg.embed_dim, swin_cfg.id_embed_dim)
        self.mod_embed = mod_embed

    def forward(self, x: torch.Tensor, mod_id: torch.Tensor) -> list[torch.Tensor]:
        """*x*: (B, C, H, W).  *mod_id*: (B,) int tensor."""
        cond = self.mod_embed(mod_id)  # (B, id_embed_dim)

        # Patch embedding (no LayerNorm — CLN handles it)
        x = self.swin.patch_embed(x)   # (B, embed_dim, H', W')

        # CLN injection instead of patch_norm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, L, C)
        x = self.cln(x, cond)                             # CLN
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)     # back to (B, C, H, W)

        # Pass through rest of Swin (skip patch_embed since already done)
        Wh, Ww = H, W
        if self.swin.ape:
            ape = nn.functional.interpolate(
                self.swin.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + ape).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.swin.pos_drop(x)

        outs = []
        for i in range(self.swin.num_layers):
            x_out, H, W, x, Wh, Ww = self.swin.layers[i](x, Wh, Ww)
            if i in self.swin.out_indices:
                norm = getattr(self.swin, f'norm{i}')
                x_out = norm(x_out)
                out = x_out.view(-1, H, W, self.swin.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


# =============================================================================
# TransCUT — full model
# =============================================================================


class TransCUT(nn.Module):
    """Full TransCUT model for training and inference.

    Encoder: Swin-Transformer + CLN (source modality injection).
    Decoder: Lightweight CNN + AdaIN (target modality injection).
    Losses: GAN + PatchNCE (same as original CUT).

    Usage::

        config = TransCUTConfig(num_modalities=6)
        model = TransCUT(config)

        # --- Training iteration ---
        fake = model(source_img, src_id=0, tgt_id=1)
        loss_G, loss_D = model.compute_losses(source_img, target_img, src_id=0, tgt_id=1)
    """

    def __init__(self, config: TransCUTConfig):
        super().__init__()
        if config.nce_fake_modality not in {"source", "target"}:
            raise ValueError("nce_fake_modality must be 'source' or 'target'")
        if config.d_update_freq < 1:
            raise ValueError("d_update_freq must be at least 1")
        if config.num_modalities < 2:
            raise ValueError("TransCUT requires at least two modalities")
        if config.decoder_variant not in {
            "legacy", "highres_content", "fullres_residual",
        }:
            raise ValueError(
                "decoder_variant must be 'legacy', 'highres_content' or "
                "'fullres_residual'"
            )
        if (
            config.lambda_paired < 0.0 or config.lambda_cycle < 0.0
            or config.lambda_D_mismatch < 0.0
        ):
            raise ValueError(
                "paired, cycle and discriminator-mismatch weights must be non-negative"
            )
        if config.lambda_cycle > 0.0:
            warnings.warn(
                "lambda_cycle enables experimental cycle consistency and assumes "
                "the cross-domain mapping is approximately invertible; it is not "
                "appropriate for the default HEMIT H&E/marker experiment",
                RuntimeWarning,
            )
        self.config = config
        device = torch.device(f"cuda:{config.gpu_ids[0]}" if config.gpu_ids else "cpu")
        self.device = device

        # --- Modality ID embedding (shared by encoder and decoder) ---
        self.mod_embed = ModalityIDEmbedding(config.num_modalities, config.id_embed_dim)
        self.modality_names = [f"modality_{index}" for index in range(config.num_modalities)]

        # --- Encoder ---
        self.encoder = SwinEncoderWithCLN(config, self.mod_embed)

        # --- Decoder ---
        decoder_kwargs = {
            "embed_dim": config.embed_dim,
            "output_nc": config.output_nc,
            "style_dim": config.decoder_style_dim,
            "n_layers": len(config.out_indices),
            "output_scale": config.patch_size,
        }
        if config.decoder_variant == "highres_content":
            self.decoder = HighResolutionContentDecoder(
                input_nc=config.input_nc,
                **decoder_kwargs,
            )
        elif config.decoder_variant == "fullres_residual":
            self.decoder = FullResolutionResidualDecoder(
                input_nc=config.input_nc,
                **decoder_kwargs,
            )
        else:
            self.decoder = TransCUTDecoder(**decoder_kwargs)
        # Target-ID → style vector for decoder AdaIN
        self.style_embed = nn.Linear(config.id_embed_dim, config.decoder_style_dim)

        # --- NCE projection head ---
        self.netF = define_F(
            config.input_nc, "mlp_sample", "instance", False,
            "normal", 0.02, False, config.gpu_ids, config,
        )

        # --- Discriminator ---
        self.netD = ConditionalPatchDiscriminator(
            config.output_nc, config.num_modalities,
            config.ndf, config.n_layers_D,
        )
        init_net(self.netD, "normal", 0.02, config.gpu_ids)

        # --- Losses ---
        self.criterionGAN = GANLoss(config.gan_mode).to(device)
        self.nce_layers = list(config.out_indices)
        self.criterionNCE = nn.ModuleList([
            PatchNCELoss(nce_T=config.nce_T, batch_size=1).to(device)
            for _ in self.nce_layers
        ])
        self.criterionIdt = nn.L1Loss().to(device)

        # --- Optimisers ---
        self.optimizer_G = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters())
            + list(self.style_embed.parameters()),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=config.lr_D or config.lr * 0.5,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_F: torch.optim.Adam | None = None  # lazy-init (PatchSampleF MLPs)
        self._F_initialized = False
        self._optimization_step = 0
        self._last_batch_diagnostics: dict[str, torch.Tensor] = {}

        self.to(device)

    def set_modality_names(self, names: list[str]) -> None:
        """Attach the stable ID-to-name mapping saved with checkpoints."""
        if len(names) != self.config.num_modalities:
            raise ValueError(
                f"expected {self.config.num_modalities} modality names, got {len(names)}"
            )
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("modality names must be non-empty and unique")
        self.modality_names = list(names)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, src_id: int | torch.Tensor,
                tgt_id: int | torch.Tensor) -> torch.Tensor:
        """Translate *x* from source modality to target modality.

        Parameters
        ----------
        x : (B, C, H, W)  Source image.
        src_id : int | (B,) int tensor  Source modality index.
        tgt_id : int | (B,) int tensor  Target modality index.

        Returns
        -------
        (B, C, H, W)  Translated image.
        """
        if isinstance(src_id, int):
            src_id = torch.full((x.size(0),), src_id, dtype=torch.long, device=x.device)
        if isinstance(tgt_id, int):
            tgt_id = torch.full((x.size(0),), tgt_id, dtype=torch.long, device=x.device)

        feats = self.encoder(x, src_id)
        style = self.style_embed(self.mod_embed(tgt_id))
        return self.decoder(feats, style, source=x)

    def encode(self, x: torch.Tensor, src_id: int | torch.Tensor,
               nce_layers: list[int] | None = None) -> list[torch.Tensor]:
        """Extract features for NCE loss (encode_only mode)."""
        if isinstance(src_id, int):
            src_id = torch.full((x.size(0),), src_id, dtype=torch.long, device=x.device)
        feats = self.encoder(x, src_id)
        if nce_layers is not None:
            feats = [feats[i] for i in nce_layers if i < len(feats)]
        return feats

    def initialize_netF(self, real_src: torch.Tensor,
                        src_id: int | torch.Tensor) -> None:
        """Materialise PatchSampleF's lazy MLPs without updating parameters."""
        if self._F_initialized or self.config.lambda_NCE <= 0.0:
            return
        with torch.no_grad():
            feats = self.encode(real_src, src_id)
            self.netF(feats, self.config.num_patches, None)
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(), lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
        )
        self._F_initialized = True

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def clear_batch_diagnostics(self) -> None:
        """Discard per-sample diagnostics from the previous optimization step."""
        self._last_batch_diagnostics.clear()

    def batch_diagnostics(self) -> dict[str, torch.Tensor]:
        """Detached per-sample metrics from the latest optimization step."""
        return dict(self._last_batch_diagnostics)

    def _gan_loss_per_sample(
        self, prediction: torch.Tensor, target_is_real: bool,
    ) -> torch.Tensor:
        """Reduce PatchGAN locations while preserving the batch dimension."""
        target = self.criterionGAN.get_target_tensor(
            prediction, target_is_real,
        )
        if self.config.gan_mode == "lsgan":
            loss = (prediction - target).square()
        elif self.config.gan_mode == "vanilla":
            loss = F.binary_cross_entropy_with_logits(
                prediction, target, reduction="none",
            )
        elif self.config.gan_mode == "wgangp":
            loss = -prediction if target_is_real else prediction
        elif self.config.gan_mode == "nonsaturating":
            loss = (
                F.softplus(-prediction)
                if target_is_real else F.softplus(prediction)
            )
        else:
            raise RuntimeError(f"unsupported GAN mode: {self.config.gan_mode}")
        return loss.flatten(1).mean(dim=1)

    @staticmethod
    def _image_statistics(
        image: torch.Tensor, dark_threshold: float = 5.0 / 255.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample mean, std and dark-pixel fraction in [0, 1]."""
        unit = image.detach().clamp(-1, 1).add(1).div(2)
        flat = unit.flatten(1)
        return (
            flat.mean(dim=1),
            flat.std(dim=1, unbiased=False),
            (flat < dark_threshold).float().mean(dim=1),
        )

    @staticmethod
    def multiscale_gradient_loss(
        generated: torch.Tensor, source: torch.Tensor,
        scales: tuple[int, ...] = (1, 2, 4),
    ) -> torch.Tensor:
        """Compare edge magnitudes without requiring equal intensities.

        This loss penalises generated/deleted boundaries at several spatial
        scales.  Absolute finite differences make it tolerant to contrast
        reversal between fluorescence and bright-field domains.
        """
        if generated.shape[-2:] != source.shape[-2:]:
            generated = F.interpolate(
                generated, size=source.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        losses = []
        for scale in scales:
            if scale > 1:
                gen = F.avg_pool2d(generated, scale, stride=scale)
                src = F.avg_pool2d(source, scale, stride=scale)
            else:
                gen, src = generated, source
            gen_dx = torch.abs(gen[..., :, 1:] - gen[..., :, :-1])
            src_dx = torch.abs(src[..., :, 1:] - src[..., :, :-1])
            gen_dy = torch.abs(gen[..., 1:, :] - gen[..., :-1, :])
            src_dy = torch.abs(src[..., 1:, :] - src[..., :-1, :])
            losses.append(F.l1_loss(gen_dx, src_dx) + F.l1_loss(gen_dy, src_dy))
        return sum(losses) / (2.0 * len(losses))

    def compute_G_loss_components(
        self, real_src: torch.Tensor, real_tgt: torch.Tensor,
        src_id: int | torch.Tensor, tgt_id: int | torch.Tensor,
        fake: torch.Tensor | None = None,
        paired_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if fake is None:
            fake = self.forward(real_src, src_id, tgt_id)

        # GAN loss (discriminator works on decoded image)
        pred_fake = self.netD(fake, tgt_id)
        gan_per_sample = self._gan_loss_per_sample(pred_fake, True)
        loss_G_GAN = gan_per_sample.mean()

        # NCE loss: compare features of fake vs real through the SAME encoder.
        # Upsample fake to match real_src spatial size before encoding.
        if fake.shape[-2:] != real_src.shape[-2:]:
            fake_up = nn.functional.interpolate(fake, size=real_src.shape[-2:],
                                                mode='bilinear', align_corners=False)
        else:
            fake_up = fake

        fake_mod_id = tgt_id if self.config.nce_fake_modality == "target" else src_id
        feat_q_raw = self.encode(fake_up, fake_mod_id)
        feat_k_raw = self.encode(real_src, src_id)
        feat_k_pool, sample_ids = self.netF(feat_k_raw, self.config.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q_raw, self.config.num_patches, sample_ids)

        for criterion in self.criterionNCE:
            criterion.batch_size = real_src.size(0)

        nce_per_layer = [
            crit(f_q, f_k).view(real_src.size(0), -1).mean(dim=1)
            for f_q, f_k, crit in zip(
                feat_q_pool, feat_k_pool, self.criterionNCE,
            )
        ]
        nce_per_sample = (
            torch.stack(nce_per_layer, dim=0).mean(dim=0)
            * self.config.lambda_NCE
        )
        loss_NCE = nce_per_sample.mean()

        # Lightweight cycle consistency.  The generated target image has
        # already been encoded for PatchNCE, so reuse those features and only
        # decode once more with the source modality style.  This gives a
        # direct, modality-agnostic content-retention constraint without a
        # second encoder pass or any geometric augmentation.
        cycle_per_sample = torch.zeros(
            real_src.size(0), device=real_src.device,
        )
        loss_cycle = torch.tensor(0.0, device=real_src.device)
        if self.config.lambda_cycle > 0.0:
            cycle_src_id = src_id
            if isinstance(cycle_src_id, int):
                cycle_src_id = torch.full(
                    (real_src.size(0),), cycle_src_id,
                    dtype=torch.long, device=real_src.device,
                )
            source_style = self.style_embed(self.mod_embed(cycle_src_id))
            reconstructed_src = self.decoder(
                feat_q_raw, source_style, source=fake_up,
            )
            if reconstructed_src.shape[-2:] != real_src.shape[-2:]:
                reconstructed_src = F.interpolate(
                    reconstructed_src, size=real_src.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            cycle_per_sample = (
                (reconstructed_src - real_src).abs().flatten(1).mean(dim=1)
                * self.config.lambda_cycle
            )
            loss_cycle = cycle_per_sample.mean()

        # Identity loss
        loss_idt = torch.tensor(0.0, device=real_src.device)
        identity_per_sample = torch.zeros(
            real_src.size(0), device=real_src.device,
        )
        if self.config.nce_idt:
            idt = self.forward(real_tgt, tgt_id, tgt_id)
            idt_up = nn.functional.interpolate(idt, size=real_tgt.shape[-2:],
                                               mode='bilinear', align_corners=False)
            identity_per_sample = (
                (idt_up - real_tgt).abs().flatten(1).mean(dim=1)
                * self.config.lambda_identity
            )
            loss_idt = identity_per_sample.mean()

        loss_structure = torch.tensor(0.0, device=real_src.device)
        if self.config.lambda_structure > 0.0:
            loss_structure = (
                self.multiscale_gradient_loss(fake_up, real_src)
                * self.config.lambda_structure
            )

        # Optional aligned-target anchor. ``paired_mask`` permits a small
        # fraction of aligned samples inside an otherwise unpaired batch.  A
        # missing mask retains the paired-only control's historical behaviour.
        paired_per_sample = torch.zeros(
            real_src.size(0), device=real_src.device,
        )
        if paired_mask is None:
            paired_mask = torch.ones_like(paired_per_sample)
        else:
            paired_mask = paired_mask.to(
                device=real_src.device, dtype=paired_per_sample.dtype,
            ).flatten()
            if paired_mask.numel() != real_src.size(0):
                raise ValueError("paired_mask must contain one value per sample")
            if torch.any((paired_mask < 0) | (paired_mask > 1)):
                raise ValueError("paired_mask values must be in [0, 1]")
        loss_paired = torch.tensor(0.0, device=real_src.device)
        if self.config.lambda_paired > 0.0:
            if fake_up.shape != real_tgt.shape:
                raise RuntimeError(
                    "paired reconstruction requires generated and target images "
                    "with identical shapes"
                )
            paired_per_sample = (
                (fake_up - real_tgt).abs().flatten(1).mean(dim=1)
                * self.config.lambda_paired
                * paired_mask
            )
            loss_paired = paired_per_sample.mean()

        weighted_gan = loss_G_GAN * self.config.lambda_GAN
        fake_mean, fake_std, fake_dark = self._image_statistics(fake_up)
        real_mean, real_std, real_dark = self._image_statistics(real_tgt)
        self._last_batch_diagnostics.update({
            "G_GAN": (
                gan_per_sample * self.config.lambda_GAN
            ).detach(),
            "NCE": nce_per_sample.detach(),
            "identity": identity_per_sample.detach(),
            "paired": paired_per_sample.detach(),
            "paired_anchor": paired_mask.detach(),
            "cycle": cycle_per_sample.detach(),
            "fake_mean": fake_mean,
            "fake_std": fake_std,
            "fake_dark_fraction": fake_dark,
            "real_mean": real_mean,
            "real_std": real_std,
            "real_dark_fraction": real_dark,
        })
        return {
            "G": (
                weighted_gan + loss_NCE + loss_idt
                + loss_paired + loss_cycle + loss_structure
            ),
            "G_GAN": weighted_gan,
            "NCE": loss_NCE,
            "identity": loss_idt,
            "paired": loss_paired,
            "cycle": loss_cycle,
            "structure": loss_structure,
        }

    def compute_G_loss(self, real_src: torch.Tensor, real_tgt: torch.Tensor,
                       src_id: int | torch.Tensor,
                       tgt_id: int | torch.Tensor,
                       paired_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Backward-compatible total generator loss."""
        return self.compute_G_loss_components(
            real_src, real_tgt, src_id, tgt_id, paired_mask=paired_mask,
        )["G"]

    def compute_D_loss_components(
        self, fake: torch.Tensor, real: torch.Tensor,
        target_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_fake = self.netD(fake.detach(), target_id)
        d_fake_per_sample = self._gan_loss_per_sample(pred_fake, False)
        loss_D_fake = d_fake_per_sample.mean()
        pred_real = self.netD(real, target_id)
        d_real_per_sample = self._gan_loss_per_sample(pred_real, True)
        loss_D_real = d_real_per_sample.mean()

        # A conditional discriminator can otherwise minimise its loss while
        # ignoring the modality ID, because ordinary GAN training only shows
        # correctly-labelled real images.  Treat a real image paired with a
        # different modality ID as an additional negative.  This is generic
        # for any number or meaning of modalities and adds no per-domain head.
        d_mismatch_per_sample = torch.zeros_like(d_real_per_sample)
        loss_D_mismatch = torch.tensor(0.0, device=real.device)
        if self.config.lambda_D_mismatch > 0.0:
            offsets = torch.randint(
                1, self.config.num_modalities, target_id.shape,
                device=target_id.device,
            )
            mismatched_id = (target_id + offsets) % self.config.num_modalities
            pred_mismatch = self.netD(real, mismatched_id)
            d_mismatch_per_sample = (
                self._gan_loss_per_sample(pred_mismatch, False)
                * self.config.lambda_D_mismatch
            )
            loss_D_mismatch = d_mismatch_per_sample.mean()
        d_denominator = 2.0 + self.config.lambda_D_mismatch
        d_per_sample = (
            d_fake_per_sample + d_real_per_sample + d_mismatch_per_sample
        ) / d_denominator
        self._last_batch_diagnostics.update({
            "D": d_per_sample.detach(),
            "D_fake": d_fake_per_sample.detach(),
            "D_real": d_real_per_sample.detach(),
            "D_mismatch": d_mismatch_per_sample.detach(),
        })
        return {
            "D": (
                loss_D_fake + loss_D_real + loss_D_mismatch
            ) / d_denominator,
            "D_fake": loss_D_fake,
            "D_real": loss_D_real,
            "D_mismatch": loss_D_mismatch,
        }

    def compute_D_loss(self, fake: torch.Tensor, real: torch.Tensor,
                       target_id: torch.Tensor) -> torch.Tensor:
        """Backward-compatible total discriminator loss."""
        return self.compute_D_loss_components(fake, real, target_id)["D"]

    # ------------------------------------------------------------------
    # Optimization step
    # ------------------------------------------------------------------

    def optimize_parameters(self, real_src: torch.Tensor,
                            real_tgt: torch.Tensor,
                            src_id: int | torch.Tensor,
                            tgt_id: int | torch.Tensor,
                            paired_mask: torch.Tensor | None = None,
                            ) -> dict[str, float]:
        self.clear_batch_diagnostics()
        # Generate fake
        fake = self.forward(real_src, src_id, tgt_id)

        # --- Update D ---
        update_discriminator = self._optimization_step % self.config.d_update_freq == 0
        if update_discriminator:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            d_losses = self.compute_D_loss_components(fake, real_tgt, tgt_id)
            d_losses["D"].backward()
            self.optimizer_D.step()
        else:
            self.set_requires_grad(self.netD, False)
            with torch.no_grad():
                d_losses = self.compute_D_loss_components(fake, real_tgt, tgt_id)

        # --- Update G ---
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()
        g_losses = self.compute_G_loss_components(
            real_src, real_tgt, src_id, tgt_id, fake=fake,
            paired_mask=paired_mask,
        )
        g_losses["G"].backward()

        # Lazy-init netF optimizer (PatchSampleF creates MLPs during first forward)
        if not self._F_initialized and self.config.lambda_NCE > 0:
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2),
            )
            self._F_initialized = True

        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()
        self._optimization_step += 1

        losses = {**g_losses, **d_losses}
        return {name: float(value.detach()) for name, value in losses.items()}

    @staticmethod
    def set_requires_grad(net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def update_learning_rate(self, completed_epoch: int) -> None:
        """Keep LR constant for ``n_epochs``, then linearly decay."""
        decay_epochs = max(1, self.config.n_epochs_decay)
        decay_progress = max(0, completed_epoch - self.config.n_epochs)
        decay_frac = max(0.0, 1.0 - decay_progress / decay_epochs)
        new_lr_g = max(self.config.lr * decay_frac, 1e-7)
        base_lr_d = self.config.lr_D or self.config.lr * 0.5
        new_lr_d = max(base_lr_d * decay_frac, 1e-7)
        for pg in self.optimizer_G.param_groups:
            pg["lr"] = new_lr_g
        for pg in self.optimizer_D.param_groups:
            pg["lr"] = new_lr_d
        if self.optimizer_F is not None:
            for pg in self.optimizer_F.param_groups:
                pg["lr"] = new_lr_g
        self._decay_step = decay_progress

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.training_state(), path)

    def training_state(self) -> dict:
        return {
            "format_version": 3,
            "model_type": "TransCUT",
            "config": vars(self.config).copy(),
            "modality_names": list(self.modality_names),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mod_embed": self.mod_embed.state_dict(),
            "style_embed": self.style_embed.state_dict(),
            "netD": self.netD.state_dict(),
            "netF": self.netF.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "optimizer_F": self.optimizer_F.state_dict() if self.optimizer_F else None,
            "decay_step": getattr(self, "_decay_step", 0),
            "optimization_step": self._optimization_step,
        }

    def load(self, path: str, allow_modality_expansion: bool = False) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.load_training_state(
            ck, load_optimizers=False,
            allow_modality_expansion=allow_modality_expansion,
        )

    def load_training_state(
        self, ck: dict, load_optimizers: bool = True,
        allow_modality_expansion: bool = False,
    ) -> bool:
        """Restore a checkpoint and optionally expand its modality embedding.

        When expanding, all shared network weights are restored, old modality
        rows are copied by stable name, and rows for new modalities keep their
        random initialization. Optimizer state is intentionally not restored
        because Adam tensors for the embedding have the old shape.

        Returns ``True`` when an embedding expansion was performed.
        """
        saved_variant = ck.get("config", {}).get("decoder_variant", "legacy")
        if saved_variant != self.config.decoder_variant:
            raise RuntimeError(
                "checkpoint decoder variant "
                f"'{saved_variant}' does not match current variant "
                f"'{self.config.decoder_variant}'"
            )
        old_embedding = ck["mod_embed"]["embedding.weight"]
        current_embedding = self.mod_embed.embedding.weight.detach().clone()
        old_names = ck.get("modality_names")
        registry_changed = old_names is not None and list(old_names) != self.modality_names
        expanded = old_embedding.shape != current_embedding.shape or registry_changed
        if expanded and not allow_modality_expansion:
            raise RuntimeError(
                "checkpoint modality embedding shape differs from the current model; "
                "enable modality expansion to initialize new modality rows"
            )
        if old_embedding.shape[1] != current_embedding.shape[1]:
            raise RuntimeError("checkpoint modality embedding dimension is incompatible")

        if expanded:
            new_names = self.modality_names
            if old_names is None:
                if old_embedding.shape[0] > current_embedding.shape[0]:
                    raise RuntimeError("cannot shrink a legacy checkpoint modality embedding")
                row_pairs = [(index, index) for index in range(old_embedding.shape[0])]
            else:
                if len(old_names) != old_embedding.shape[0]:
                    raise RuntimeError("checkpoint modality_names do not match its embedding")
                new_by_name = {name: index for index, name in enumerate(new_names)}
                missing = [name for name in old_names if name not in new_by_name]
                if missing:
                    raise RuntimeError(
                        f"current registry is missing checkpoint modalities: {missing}"
                    )
                row_pairs = [
                    (old_index, new_by_name[name])
                    for old_index, name in enumerate(old_names)
                ]
            for old_index, new_index in row_pairs:
                current_embedding[new_index].copy_(old_embedding[old_index])
        else:
            row_pairs = [(index, index) for index in range(old_embedding.shape[0])]
            current_embedding.copy_(old_embedding)

        encoder_state = dict(ck["encoder"])
        encoder_state["mod_embed.embedding.weight"] = current_embedding
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(ck["decoder"], strict=True)
        self.mod_embed.load_state_dict(
            {"embedding.weight": current_embedding}, strict=True,
        )
        self.style_embed.load_state_dict(ck["style_embed"], strict=True)
        if "netD" in ck:
            discriminator_state = dict(ck["netD"])
            discriminator_key = "modality_embedding.weight"
            if discriminator_key in discriminator_state:
                old_d_embedding = discriminator_state[discriminator_key]
                current_d_embedding = (
                    self.netD.modality_embedding.weight.detach().clone()
                )
                if old_d_embedding.shape[1] != current_d_embedding.shape[1]:
                    raise RuntimeError(
                        "checkpoint discriminator modality embedding is incompatible"
                    )
                for old_index, new_index in row_pairs:
                    current_d_embedding[new_index].copy_(old_d_embedding[old_index])
                discriminator_state[discriminator_key] = current_d_embedding
                self.netD.load_state_dict(discriminator_state, strict=True)
            elif allow_modality_expansion:
                warnings.warn(
                    "legacy unconditional discriminator was not restored; "
                    "the conditional discriminator remains newly initialized",
                    RuntimeWarning,
                )
            else:
                self.netD.load_state_dict(discriminator_state, strict=True)
        # netF is lazy; callers must initialise it before restoring a non-empty state.
        if ck.get("netF"):
            if not self._F_initialized:
                raise RuntimeError("TransCUT netF must be initialized before checkpoint restore")
            self.netF.load_state_dict(ck["netF"], strict=True)
        if load_optimizers and expanded:
            raise RuntimeError(
                "optimizer state cannot be restored while expanding modalities; "
                "load weights for a new fine-tuning run instead"
            )
        if load_optimizers:
            if "optimizer_G" in ck:
                self.optimizer_G.load_state_dict(ck["optimizer_G"])
            if "optimizer_D" in ck:
                self.optimizer_D.load_state_dict(ck["optimizer_D"])
            if ck.get("optimizer_F") is not None:
                if self.optimizer_F is None:
                    raise RuntimeError("TransCUT optimizer_F is not initialized")
                self.optimizer_F.load_state_dict(ck["optimizer_F"])
        self._decay_step = ck.get("decay_step", 0)
        self._optimization_step = ck.get("optimization_step", 0)
        return expanded

    def get_encoder_state(self) -> dict[str, torch.Tensor]:
        """Return encoder weights for sharing with TransMorph."""
        return {
            "swin": self.encoder.swin.state_dict(),
            "mod_embed": self.mod_embed.state_dict(),
        }
