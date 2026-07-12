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

from crossreg.models.swin_transformer import SwinTransformer
from crossreg.translation.cut.networks import NLayerDiscriminator, GANLoss, define_F, init_net
from crossreg.translation.cut.patchnce import PatchNCELoss

from .cln_adain import ModalityIDEmbedding, CLN2d
from .decoder import TransCUTDecoder


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

    # --- Discriminator ---
    ndf: int = 64
    n_layers_D: int = 3
    gan_mode: str = "lsgan"

    # --- NCE ---
    netF_nc: int = 256
    nce_T: float = 0.07
    num_patches: int = 256

    # --- Loss weights ---
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    nce_idt: bool = True

    # --- Training ---
    lr: float = 2e-4
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
        self.config = config
        device = torch.device(f"cuda:{config.gpu_ids[0]}" if config.gpu_ids else "cpu")
        self.device = device

        # --- Modality ID embedding (shared by encoder and decoder) ---
        self.mod_embed = ModalityIDEmbedding(config.num_modalities, config.id_embed_dim)

        # --- Encoder ---
        self.encoder = SwinEncoderWithCLN(config, self.mod_embed)

        # --- Decoder ---
        self.decoder = TransCUTDecoder(
            embed_dim=config.embed_dim,
            output_nc=config.output_nc,
            style_dim=config.decoder_style_dim,
            n_layers=len(config.out_indices),
        )
        # Target-ID → style vector for decoder AdaIN
        self.style_embed = nn.Linear(config.id_embed_dim, config.decoder_style_dim)

        # --- NCE projection head ---
        self.netF = define_F(
            config.input_nc, "mlp_sample", "instance", False,
            "normal", 0.02, False, config.gpu_ids, config,
        )

        # --- Discriminator ---
        self.netD = NLayerDiscriminator(
            config.output_nc, config.ndf, config.n_layers_D
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
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config.lr, betas=(config.beta1, config.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
        )
        self.optimizer_F: torch.optim.Adam | None = None  # lazy-init (PatchSampleF MLPs)
        self._F_initialized = False

        self.to(device)

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
        return self.decoder(feats, style)

    def encode(self, x: torch.Tensor, src_id: int | torch.Tensor,
               nce_layers: list[int] | None = None) -> list[torch.Tensor]:
        """Extract features for NCE loss (encode_only mode)."""
        if isinstance(src_id, int):
            src_id = torch.full((x.size(0),), src_id, dtype=torch.long, device=x.device)
        feats = self.encoder(x, src_id)
        if nce_layers is not None:
            feats = [feats[i] for i in nce_layers if i < len(feats)]
        return feats

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_G_loss(self, real_src: torch.Tensor, real_tgt: torch.Tensor,
                       src_id: int, tgt_id: int) -> torch.Tensor:
        fake = self.forward(real_src, src_id, tgt_id)

        # GAN loss (discriminator works on decoded image)
        pred_fake = self.netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # NCE loss: compare features of fake vs real through the SAME encoder.
        # Upsample fake to match real_src spatial size before encoding.
        if fake.shape[-2:] != real_src.shape[-2:]:
            fake_up = nn.functional.interpolate(fake, size=real_src.shape[-2:],
                                                mode='bilinear', align_corners=False)
        else:
            fake_up = fake

        feat_q_raw = self.encode(fake_up, src_id)
        feat_k_raw = self.encode(real_src, src_id)
        feat_k_pool, sample_ids = self.netF(feat_k_raw, self.config.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q_raw, self.config.num_patches, sample_ids)

        loss_NCE = sum(
            crit(f_q, f_k).mean() * self.config.lambda_NCE
            for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE)
        ) / len(self.criterionNCE)

        # Identity loss
        loss_idt = torch.tensor(0.0, device=real_src.device)
        if self.config.nce_idt:
            idt = self.forward(real_tgt, tgt_id, tgt_id)
            idt_up = nn.functional.interpolate(idt, size=real_tgt.shape[-2:],
                                               mode='bilinear', align_corners=False)
            loss_idt = self.criterionIdt(idt_up, real_tgt) * self.config.lambda_NCE

        return loss_G_GAN * self.config.lambda_GAN + loss_NCE + loss_idt

    def compute_D_loss(self, fake: torch.Tensor,
                       real: torch.Tensor) -> torch.Tensor:
        pred_fake = self.netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_real = self.netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        return (loss_D_fake + loss_D_real) * 0.5

    # ------------------------------------------------------------------
    # Optimization step
    # ------------------------------------------------------------------

    def optimize_parameters(self, real_src: torch.Tensor,
                            real_tgt: torch.Tensor,
                            src_id: int, tgt_id: int) -> dict[str, float]:
        # Generate fake
        fake = self.forward(real_src, src_id, tgt_id)

        # --- Update D ---
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(fake, real_tgt)
        loss_D.backward()
        self.optimizer_D.step()

        # --- Update G ---
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        loss_G = self.compute_G_loss(real_src, real_tgt, src_id, tgt_id)
        loss_G.backward()

        # Lazy-init netF optimizer (PatchSampleF creates MLPs during first forward)
        if not self._F_initialized and self.config.lambda_NCE > 0:
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2),
            )
            self._F_initialized = True

        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()

        return {"G": float(loss_G.detach()), "D": float(loss_D.detach())}

    @staticmethod
    def set_requires_grad(net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def update_learning_rate(self) -> None:
        if not hasattr(self, "_decay_step"):
            self._decay_step = 0
        decay_frac = max(0.0, 1.0 - self._decay_step / max(1, self.config.n_epochs_decay))
        new_lr = self.config.lr * decay_frac
        for opt in [self.optimizer_G, self.optimizer_D]:
            for pg in opt.param_groups:
                pg["lr"] = max(new_lr, 1e-7)
        if self.optimizer_F is not None:
            for pg in self.optimizer_F.param_groups:
                pg["lr"] = max(new_lr, 1e-7)
        self._decay_step += 1

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mod_embed": self.mod_embed.state_dict(),
            "style_embed": self.style_embed.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ck["encoder"])
        self.decoder.load_state_dict(ck["decoder"])
        self.mod_embed.load_state_dict(ck["mod_embed"])
        self.style_embed.load_state_dict(ck["style_embed"])

    def get_encoder_state(self) -> dict[str, torch.Tensor]:
        """Return encoder weights for sharing with TransMorph."""
        return {
            "swin": self.encoder.swin.state_dict(),
            "mod_embed": self.mod_embed.state_dict(),
        }
