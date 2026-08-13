"""
Simplified CUT model — decoupled from argparse / BaseModel.

Provides:
- CUTWrapper: full training model (G + D + F + NCE + GAN losses)
- CUTInference: lightweight generator-only wrapper for translation inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field

from .networks import define_G, define_D, define_F, GANLoss, init_net
from .patchnce import PatchNCELoss


# =============================================================================
# Config
# =============================================================================


@dataclass
class CUTConfig:
    """Configuration for CUT model, replacing the argparse-based options."""

    # I/O
    input_nc: int = 3
    output_nc: int = 3

    # Generator
    netG: str = "resnet_9blocks"
    ngf: int = 64
    normG: str = "instance"
    no_dropout: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02
    no_antialias: bool = False
    no_antialias_up: bool = False

    # Discriminator
    netD: str = "basic"
    ndf: int = 64
    n_layers_D: int = 3
    normD: str = "instance"
    gan_mode: str = "lsgan"

    # NCE projection head
    netF: str = "mlp_sample"
    netF_nc: int = 256
    nce_layers: list[int] = field(default_factory=lambda: [0, 4, 8, 12, 16])
    nce_T: float = 0.07
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False

    # Loss weights
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    nce_idt: bool = True
    flip_equivariance: bool = False

    # Training
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"
    n_epochs: int = 200
    n_epochs_decay: int = 200
    epoch_count: int = 1
    lr_decay_iters: int = 50  # for "step" policy

    # Device
    gpu_ids: list[int] = field(default_factory=lambda: [0])

    # Direction
    direction: str = "AtoB"  # "AtoB" or "BtoA"


# =============================================================================
# CUT Training Wrapper
# =============================================================================


class CUTWrapper(nn.Module):
    """Full CUT model for training: Generator + Discriminator + NCE projection + losses.

    Usage::

        config = CUTConfig(input_nc=1, output_nc=1)
        model = CUTWrapper(config)

        # --- training iteration ---
        model.set_input({"A": real_A, "B": real_B, "A_paths": ["/path/a.png"]})
        model.optimize_parameters()
        losses = model.get_current_losses()
    """

    def __init__(self, config: CUTConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_ids[0]}" if config.gpu_ids else "cpu")
        self.isTrain = True

        # loss names (matches original CUT)
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G", "NCE"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        if config.nce_idt:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["idt_B"]
        self.model_names = ["G", "F", "D"]

        # --- networks ---
        self.netG = define_G(
            config.input_nc, config.output_nc, config.ngf, config.netG,
            config.normG, config.no_dropout, config.init_type, config.init_gain,
            config.no_antialias, config.no_antialias_up, config.gpu_ids, config,
        )

        self.netF = define_F(
            config.input_nc, config.netF, config.normG, config.no_dropout,
            config.init_type, config.init_gain, config.no_antialias, config.gpu_ids, config,
        )

        self.netD = define_D(
            config.output_nc, config.ndf, config.netD, config.n_layers_D,
            config.normD, config.init_type, config.init_gain,
            config.no_antialias, config.gpu_ids, config,
        )

        # --- losses ---
        self.criterionGAN = GANLoss(config.gan_mode).to(self.device)
        self.nce_layers = config.nce_layers
        self.criterionNCE = nn.ModuleList()
        for _ in self.nce_layers:
            self.criterionNCE.append(
                PatchNCELoss(
                    nce_T=config.nce_T,
                    batch_size=1,  # will be overridden at first forward
                    nce_includes_all_negatives_from_minibatch=config.nce_includes_all_negatives_from_minibatch,
                ).to(self.device)
            )
        self.criterionIdt = nn.L1Loss().to(self.device)

        # --- optimizers ---
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        self.optimizer_F = None
        self._F_initialized = False

        # state
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.idt_B = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_input(self, data: dict):
        AtoB = self.config.direction == "AtoB"
        self.real_A = data["A" if AtoB else "B"].to(self.device)
        self.real_B = data["B" if AtoB else "A"].to(self.device)

    def data_dependent_initialize(self, data: dict):
        """Initialize netF after first forward pass (needs feature shapes)."""
        bs = data["A"].size(0)
        self.set_input(data)
        self.forward()
        self.compute_D_loss().backward()
        self.compute_G_loss().backward()
        if self.config.lambda_NCE > 0.0:
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2)
            )
            self._F_initialized = True

    def initialize_netF(self, data: dict) -> None:
        """Materialise PatchSampleF's lazy MLPs without updating parameters."""
        if self._F_initialized or self.config.lambda_NCE <= 0.0:
            return
        self.set_input(data)
        self.forward()
        with torch.no_grad():
            feat = self.netG(self.real_A, self.nce_layers, encode_only=True)
            self.netF(feat, self.config.num_patches, None)
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(), lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
        )
        self._F_initialized = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self):
        if self.config.nce_idt and self.isTrain:
            self.real = torch.cat((self.real_A, self.real_B), dim=0)
        else:
            self.real = self.real_A

        if self.config.flip_equivariance:
            self.flipped_for_equivariance = self.isTrain and (torch.rand(1).item() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        else:
            self.flipped_for_equivariance = False

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if self.config.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def compute_D_loss(self):
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        if self.config.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.config.lambda_GAN
        else:
            self.loss_G_GAN = torch.tensor(0.0, device=self.device)

        if self.config.lambda_NCE > 0.0:
            self.loss_NCE = self._calculate_nce_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE = torch.tensor(0.0, device=self.device)

        if self.config.nce_idt and self.config.lambda_NCE > 0.0:
            self.loss_NCE_Y = self._calculate_nce_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def _calculate_nce_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        for criterion in self.criterionNCE:
            criterion.batch_size = src.size(0)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if self.config.flip_equivariance and getattr(self, "flipped_for_equivariance", False):
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.config.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.config.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.config.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G (+ F)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.optimizer_F is not None:
            self.optimizer_F.zero_grad()

        # compute_G_loss triggers netF.forward() → create_mlp() on first call,
        # so netF parameters only become non-empty inside this call.
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        # Lazy-init netF optimizer once its MLP layers exist.
        if not self._F_initialized and self.config.lambda_NCE > 0.0:
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2)
            )
            self._F_initialized = True

        self.optimizer_G.step()
        if self.optimizer_F is not None:
            self.optimizer_F.step()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def set_requires_grad(net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def update_learning_rate(self) -> None:
        """Apply linear LR decay towards zero over the decay epochs.

        LR = initial_lr * max(0, 1 - epoch_decay / n_epochs_decay)
        where epoch_decay counts from 0 after the initial ``n_epochs``.
        """
        if not hasattr(self, "_decay_step"):
            self._decay_step = 0
        old_lr = self.optimizer_G.param_groups[0]["lr"]
        decay_frac = max(0.0, 1.0 - self._decay_step / max(1, self.config.n_epochs_decay))
        new_lr = self.config.lr * decay_frac
        if new_lr < old_lr:
            for opt in [self.optimizer_G, self.optimizer_D]:
                for pg in opt.param_groups:
                    pg["lr"] = new_lr
            if self.optimizer_F is not None:
                for pg in self.optimizer_F.param_groups:
                    pg["lr"] = new_lr
        self._decay_step += 1

    def get_current_losses(self) -> dict[str, float]:
        return {name: float(getattr(self, f"loss_{name}", 0).detach()) for name in self.loss_names}

    def get_current_visuals(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name, None) for name in self.visual_names}

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def training_state(self) -> dict:
        """Return a complete, resumable CUT training checkpoint."""
        return {
            "format_version": 1,
            "model_type": "CUT",
            "config": vars(self.config).copy(),
            "netG": self.netG.state_dict(),
            "netD": self.netD.state_dict(),
            "netF": self.netF.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "optimizer_F": self.optimizer_F.state_dict() if self.optimizer_F else None,
            "decay_step": getattr(self, "_decay_step", 0),
        }

    def load_training_state(self, checkpoint: dict) -> None:
        """Strictly restore all networks and available optimiser state."""
        self.netG.load_state_dict(checkpoint["netG"], strict=True)
        self.netD.load_state_dict(checkpoint["netD"], strict=True)
        if checkpoint.get("netF"):
            if not self._F_initialized:
                raise RuntimeError("initialize_netF(data) must be called before loading netF")
            self.netF.load_state_dict(checkpoint["netF"], strict=True)
        if "optimizer_G" in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        if "optimizer_D" in checkpoint:
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        if checkpoint.get("optimizer_F") is not None:
            if self.optimizer_F is None:
                raise RuntimeError("netF optimizer is not initialized")
            self.optimizer_F.load_state_dict(checkpoint["optimizer_F"])
        self._decay_step = checkpoint.get("decay_step", 0)

    def save_networks(self, save_dir: str, epoch: int):
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name in self.model_names:
            net = getattr(self, f"net{name}")
            state = net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict()
            path = os.path.join(save_dir, f"{epoch}_net_{name}.pth")
            torch.save(state, path)

    def load_networks(self, load_dir: str, epoch: int | str):
        import os
        for name in self.model_names:
            path = os.path.join(load_dir, f"{epoch}_net_{name}.pth")
            net = getattr(self, f"net{name}")
            if isinstance(net, nn.DataParallel):
                net = net.module
            state = torch.load(path, map_location=str(self.device))
            net.load_state_dict(state)
            print(f"Loaded {path}")

    def train(self, mode=True):
        super().train(mode)
        for name in self.model_names:
            getattr(self, f"net{name}").train(mode)
        return self

    def eval(self):
        return self.train(False)


# =============================================================================
# CUT Inference (generator-only)
# =============================================================================


class CUTInference(nn.Module):
    """Lightweight generator-only wrapper for CUT translation inference.

    Usage::

        model = CUTInference(input_nc=1, output_nc=1, netG="resnet_9blocks")
        model.load_weights("/path/to/latest_net_G.pth")
        fake_B = model.translate(real_A)
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        netG: str = "resnet_9blocks",
        norm: str = "instance",
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
        gpu_ids: list[int] | None = None,
    ):
        super().__init__()
        if gpu_ids is None:
            gpu_ids = [0] if torch.cuda.is_available() else []
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.gpu_ids = gpu_ids

        norm_layer = nn.InstanceNorm2d
        from .networks import ResnetGenerator

        self.netG = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer,
            use_dropout=False, n_blocks=9 if netG == "resnet_9blocks" else 6,
            padding_type="reflect", no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
        )
        self.netG.to(self.device)
        self.netG.eval()

    def load_weights(self, checkpoint_path: str):
        """Load either a generator state_dict or a complete training checkpoint."""
        state = torch.load(checkpoint_path, map_location=str(self.device), weights_only=False)
        if isinstance(state, dict) and "netG" in state:
            state = state["netG"]
        self.netG.load_state_dict(state, strict=True)
        self.netG.to(self.device)
        self.netG.eval()
        print(f"Loaded generator weights from {checkpoint_path}")

    @torch.no_grad()
    def translate(self, image: torch.Tensor) -> torch.Tensor:
        """Translate a single image or batch from domain A to domain B.

        Args:
            image: tensor of shape (B, C, H, W) or (C, H, W).

        Returns:
            translated tensor of same shape.
        """
        single = image.dim() == 3
        if single:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        out = self.netG(image)
        if single:
            out = out.squeeze(0)
        return out

    def forward(self, x):
        return self.translate(x)
