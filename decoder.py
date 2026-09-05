```python
# decoder.py

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "The 'mamba-ssm' package is required. "
        "Please install it before running GeneCMamba."
    ) from exc


class MambaBlock(nn.Module):
    """
    Mamba-based cross-modal decoder block.

    Each block contains three main components:

    1. Mamba state-space modeling for sequential text/gene features.
    2. Cross-attention between text/gene queries and image tokens.
    3. Cross-modal gating for adaptive filtering of visual features.

    Parameters
    ----------
    dim : int
        Hidden dimension of the decoder.

    n_heads : int, optional
        Number of attention heads used in cross-attention.

    dropout : float, optional
        Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Mamba state-space modeling
        # ----------------------------------------------------

        self.norm_mamba = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim)

        # ----------------------------------------------------
        # Cross-modal attention
        # ----------------------------------------------------

        self.norm_cross = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # ----------------------------------------------------
        # Cross-modal gating mechanism
        # ----------------------------------------------------
        #
        # The gate is conditioned jointly on:
        #   1. normalized text/gene query features
        #   2. cross-attended visual features
        #
        # This enables adaptive suppression of less relevant
        # cross-modal information before residual fusion.
        # ----------------------------------------------------

        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of a Mamba decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Text/gene feature sequence with shape (B, T, D).

        context : torch.Tensor, optional
            Image token sequence with shape (B, N, D).

        Returns
        -------
        torch.Tensor
            Updated feature sequence with shape (B, T, D).
        """

        # ----------------------------------------------------
        # 1. Mamba sequence modeling
        # ----------------------------------------------------

        residual = x

        x = self.norm_mamba(x)
        x = self.mamba(x)

        x = residual + self.dropout(x)

        # ----------------------------------------------------
        # 2. Cross-modal attention and gating
        # ----------------------------------------------------

        if context is not None:
            residual = x

            query = self.norm_cross(x)

            cross_features, _ = self.cross_attn(
                query=query,
                key=context,
                value=context,
                need_weights=False
            )

            # ------------------------------------------------
            # Cross-modal gating
            #
            # gate = sigmoid(
            #     W2(
            #         SiLU(
            #             W1([query ; cross_features])
            #         )
            #     )
            # )
            #
            # gated_features = gate * cross_features
            # ------------------------------------------------

            gate_input = torch.cat(
                [query, cross_features],
                dim=-1
            )

            gate = self.gate_proj(
                gate_input
            )

            gated_cross_features = (
                cross_features * gate
            )

            x = residual + self.dropout(
                gated_cross_features
            )

        # ----------------------------------------------------
        # 3. Feed-forward network
        # ----------------------------------------------------

        residual = x

        x_norm = self.norm_ffn(x)

        x = residual + self.dropout(
            self.ffn(x_norm)
        )

        return x


class MambaFusionDecoder(nn.Module):
    """
    Cross-modal Mamba decoder for gene token prediction.

    The decoder first projects CLIP text representations into the
    decoder feature space. It then applies stacked Mamba blocks that
    integrate image information through cross-attention and adaptive
    cross-modal gating.

    Parameters
    ----------
    dim : int
        Hidden dimension of the decoder.

    depth : int
        Number of stacked Mamba blocks.

    vocab_size : int
        Size of the output vocabulary.

    text_hidden_dim : int, optional
        Hidden dimension of the CLIP text encoder.

    n_heads : int, optional
        Number of attention heads used for cross-attention.

    dropout : float, optional
        Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        vocab_size: int,
        text_hidden_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Text feature projection
        # ----------------------------------------------------

        self.text_proj = nn.Linear(
            text_hidden_dim,
            dim
        )

        # ----------------------------------------------------
        # Stacked Mamba decoder blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # ----------------------------------------------------
        # Final normalization and prediction head
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(dim)

        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=True
        )

    def forward(
        self,
        text_hidden: torch.Tensor,
        img_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode text/gene representations conditioned on image features.

        Parameters
        ----------
        text_hidden : torch.Tensor
            CLIP text hidden states with shape
            (B, T, text_hidden_dim).

        img_tokens : torch.Tensor
            Image token features with shape
            (B, N, dim).

        Returns
        -------
        torch.Tensor
            Token logits with shape
            (B, T, vocab_size).
        """

        # Project CLIP text features to decoder dimension
        x = self.text_proj(
            text_hidden
        )

        # Cross-modal Mamba decoding
        for block in self.blocks:
            x = block(
                x,
                context=img_tokens
            )

        # Final normalization and token prediction
        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
```
