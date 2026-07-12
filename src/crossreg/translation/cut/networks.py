"""
CUT network architectures.

Stripped from the official contrastive-unpaired-translation repo:
https://github.com/taesungp/contrastive-unpaired-translation

Kept: ResnetGenerator, NLayerDiscriminator, PatchSampleF, GANLoss, helpers.
Removed: StyleGAN2, UnetGenerator, G_Resnet variants, unused encoder/decoder blocks.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler

if TYPE_CHECKING:
    from .cut_model import CUTConfig


# =============================================================================
# Helper Functions
# =============================================================================


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1.0])
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    else:
        raise ValueError(f"Unsupported filter size: {filt_size}")

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        return nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        return nn.ReplicationPad2d
    elif pad_type == "zero":
        return nn.ZeroPad2d
    else:
        raise ValueError(f"Pad type [{pad_type}] not recognized")


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

        return norm_layer
    else:
        raise NotImplementedError(f"Normalization layer [{norm_type}] not found")


def get_scheduler(optimizer: torch.optim.Optimizer, config: CUTConfig) -> lr_scheduler.LRScheduler:
    """Build a learning-rate scheduler from a :class:`~.cut_model.CUTConfig`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to schedule.
    config : CUTConfig
        Configuration dataclass; uses ``lr_policy``, ``n_epochs``,
        ``n_epochs_decay``, ``epoch_count``, and ``lr_decay_iters``.

    Returns
    -------
    lr_scheduler.LRScheduler
    """
    if config.lr_policy == "linear":

        def lambda_rule(epoch: int) -> float:
            return 1.0 - max(0, epoch + config.epoch_count - config.n_epochs) / float(config.n_epochs_decay + 1)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    if config.lr_policy == "step":
        return lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)

    if config.lr_policy == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)

    if config.lr_policy == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=0)

    raise NotImplementedError(f"LR policy [{config.lr_policy}] not implemented")


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=None, debug=False, initialize_weights=True):
    if gpu_ids is None:
        gpu_ids = []
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


# =============================================================================
# Downsample / Upsample (anti-aliased)
# =============================================================================


class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [ps + pad_off for ps in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            return inp[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type="repl", filt_size=4, stride=2):
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1]
        )[:, :, 1:, 1:]
        return ret_val if self.filt_odd else ret_val[:, :, :-1, :-1]


# =============================================================================
# Normalize (L2)
# =============================================================================


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-7)


# =============================================================================
# ResNet Generator
# =============================================================================


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def _build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] not implemented")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator (used by CUT).

    Supports ``encode_only=True`` with ``layers`` for NCE feature extraction.
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
        no_antialias=False,
        no_antialias_up=False,
        opt=None,
    ):
        assert n_blocks >= 0
        super().__init__()
        self.opt = opt
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if no_antialias:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2),
                ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    Upsample(ngf * mult),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=None, encode_only=False):
        if layers is None:
            layers = []
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        return self.model(input)


# =============================================================================
# PatchGAN Discriminator
# =============================================================================


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf),
            ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """1x1 PixelGAN discriminator."""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, input):
        return self.net(input)


# =============================================================================
# GAN Loss
# =============================================================================


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp", "nonsaturating"]:
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        bs = prediction.size(0)
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            return -prediction.mean() if target_is_real else prediction.mean()
        elif self.gan_mode == "nonsaturating":
            if target_is_real:
                return F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                return F.softplus(prediction).view(bs, -1).mean(dim=1)
        return None


# =============================================================================
# PatchSampleF (projection head for NCE)
# =============================================================================


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type="normal", init_gain=0.02, nc=256, gpu_ids=None):
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids if gpu_ids is not None else []

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc))
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, f"mlp_{mlp_id}", mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
                    patch_id = torch.from_numpy(patch_id).to(device=feat.device, dtype=torch.long)
                if isinstance(patch_id, np.ndarray):
                    patch_id = torch.from_numpy(patch_id).to(device=feat.device, dtype=torch.long)
                elif patch_id.device != feat.device:
                    patch_id = patch_id.to(device=feat.device, dtype=torch.long)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, f"mlp_{feat_id}")
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


# =============================================================================
# Factory Functions
# =============================================================================


def define_G(
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    no_antialias=False,
    no_antialias_up=False,
    gpu_ids=None,
    opt=None,
):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt,
        )
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt,
        )
    elif netG == "resnet_4blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
            no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt,
        )
    elif netG == "unet_128":
        raise NotImplementedError("UNet generator not migrated; use resnet_* variants")
    elif netG == "unet_256":
        raise NotImplementedError("UNet generator not migrated; use resnet_* variants")
    elif netG == "stylegan2":
        raise NotImplementedError("StyleGAN2 generator not migrated")
    else:
        raise NotImplementedError(f"Generator [{netG}] not recognized")
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=("stylegan2" not in netG))


def define_D(
    input_nc,
    ndf,
    netD,
    n_layers_D=3,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    no_antialias=False,
    gpu_ids=None,
    opt=None,
):
    if gpu_ids is None:
        gpu_ids = []
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias)
    elif netD == "pixel":
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f"Discriminator [{netD}] not recognized")
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)


def define_F(
    input_nc,
    netF,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    no_antialias=False,
    gpu_ids=None,
    opt=None,
):
    if gpu_ids is None:
        gpu_ids = []
    if netF == "global_pool":
        raise NotImplementedError("global_pool netF not migrated")
    elif netF == "reshape":
        raise NotImplementedError("reshape netF not migrated")
    elif netF == "sample":
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == "mlp_sample":
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == "strided_conv":
        raise NotImplementedError("strided_conv netF not migrated")
    else:
        raise NotImplementedError(f"Projection model [{netF}] not recognized")
    return init_net(net, init_type, init_gain, gpu_ids)
