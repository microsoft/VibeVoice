import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel
from transformers.utils import logging

from .configuration_vibevoice import VibeVoiceDiffusionHeadConfig


logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation to input tensor."""
    return x * (1 + scale) + shift


# -----------------------------------------------------------------------------
# Timestep embedding
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Args:
        hidden_size (`int`): Size of the output embedding
        frequency_embedding_size (`int`, optional): Size of the intermediate
            frequency embedding
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            ACT2FN["silu"],
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t (`torch.Tensor`): 1-D tensor of timesteps
            dim (`int`): Output dimension
            max_period (`int`, optional): Minimum frequency
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )

        return embedding.to(t.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# -----------------------------------------------------------------------------
# Feed-forward blocks
# -----------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Standard feed-forward network with SwiGLU activation.

    Args:
        embed_dim (`int`): Input dimension
        ffn_dim (`int`): Hidden dimension
    """

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class HeadLayer(nn.Module):
    """
    Single layer in the diffusion head.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        cond_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.ffn_dim = ffn_dim

        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)

        self.adaLN_modulation = nn.Sequential(
            ACT2FN["silu"],
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_ffn * self.ffn(
            modulate(self.norm(x), shift_ffn, scale_ffn)
        )
        return x


class FinalLayer(nn.Module):
    """
    Final projection layer of the diffusion head.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        cond_size: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.norm_final = RMSNorm(
            hidden_size,
            eps=norm_eps,
            elementwise_affine=False,
        )
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

        self.adaLN_modulation = nn.Sequential(
            ACT2FN["silu"],
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -----------------------------------------------------------------------------
# Diffusion head model
# -----------------------------------------------------------------------------

class VibeVoiceDiffusionHead(PreTrainedModel):
    """
    Diffusion head model for VibeVoice.
    """

    config_class = VibeVoiceDiffusionHeadConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: VibeVoiceDiffusionHeadConfig):
        super().__init__(config)

        self.config = config
        self.cond_dim = config.hidden_size
        latent_size = config.latent_size

        self.noisy_images_proj = nn.Linear(
            latent_size,
            config.hidden_size,
            bias=False,
        )
        self.cond_proj = nn.Linear(
            config.hidden_size,
            self.cond_dim,
            bias=False,
        )
        self.t_embedder = TimestepEmbedder(self.cond_dim)

        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)

        self.layers = nn.ModuleList(
            [
                HeadLayer(
                    embed_dim=config.hidden_size,
                    ffn_dim=ffn_dim,
                    cond_dim=self.cond_dim,
                    norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.head_layers)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=config.rms_norm_eps,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)

        c = condition + t

        for layer in self.layers:
            x = layer(x, c)

        return self.final_layer(x, c)


AutoModel.register(VibeVoiceDiffusionHeadConfig, VibeVoiceDiffusionHead)

__all__ = ["VibeVoiceDiffusionHead"]
