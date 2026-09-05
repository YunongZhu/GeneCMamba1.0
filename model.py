```python
# model.py

import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPModel

from decoder import MambaFusionDecoder


class GeneCMambaModel(nn.Module):
    """
    GeneCMamba model for spatial gene-expression prediction from
    histopathology images.

    The model uses frozen CLIP vision and text encoders to extract
    multimodal representations. A Mamba-based fusion decoder then
    integrates image features with text/gene representations through
    cross-attention and cross-modal gating.

    Parameters
    ----------
    clip_model_name : str
        Hugging Face model identifier or local path to the CLIP model.

    decoder_dim : int
        Hidden dimension of the Mamba fusion decoder.

    decoder_depth : int
        Number of stacked Mamba decoder blocks.

    vocab_size : int
        Size of the tokenizer vocabulary.

    pretrained_clip : bool, optional
        Whether to initialize CLIP with pretrained weights.

    pretrained_path : str or None, optional
        Optional path to a previously saved GeneCMamba checkpoint.
    """

    def __init__(
        self,
        clip_model_name,
        decoder_dim,
        decoder_depth,
        vocab_size,
        pretrained_clip=True,
        pretrained_path=None
    ):
        super().__init__()

        # ----------------------------------------------------
        # CLIP backbone
        # ----------------------------------------------------

        if pretrained_clip:
            self.clip = CLIPModel.from_pretrained(
                clip_model_name
            )
        else:
            clip_config = CLIPConfig.from_pretrained(
                clip_model_name
            )

            self.clip = CLIPModel(
                clip_config
            )

        # Freeze CLIP encoder parameters
        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        # ----------------------------------------------------
        # Mamba-based cross-modal decoder
        # ----------------------------------------------------

        text_hidden_dim = (
            self.clip.config.text_config.hidden_size
        )

        self.decoder = MambaFusionDecoder(
            dim=decoder_dim,
            depth=decoder_depth,
            vocab_size=vocab_size,
            text_hidden_dim=text_hidden_dim
        )

        # ----------------------------------------------------
        # Optional checkpoint initialization
        # ----------------------------------------------------

        if pretrained_path is not None:
            self.load_checkpoint(
                pretrained_path
            )

    def load_checkpoint(
        self,
        checkpoint_path
    ):
        """
        Load model parameters from a checkpoint.

        Both complete checkpoint dictionaries containing
        ``model_state`` and raw PyTorch state dictionaries are
        supported.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state"
            ]
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = (
            self.load_state_dict(
                state_dict,
                strict=False
            )
        )

        if missing_keys:
            print(
                "[Checkpoint] Missing keys:",
                missing_keys
            )

        if unexpected_keys:
            print(
                "[Checkpoint] Unexpected keys:",
                unexpected_keys
            )

    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):
        """
        Perform a forward pass through GeneCMamba.

        Parameters
        ----------
        images : torch.Tensor
            Histopathology image patches with shape
            (B, C, H, W).

        input_ids : torch.Tensor
            Tokenized gene-expression captions with shape
            (B, T).

        attention_mask : torch.Tensor
            Text attention mask with shape
            (B, T).

        Returns
        -------
        torch.Tensor
            Predicted token logits with shape
            (B, T, vocab_size).
        """

        # ----------------------------------------------------
        # Image feature extraction
        # ----------------------------------------------------

        vision_outputs = self.clip.vision_model(
            pixel_values=images
        )

        image_tokens = (
            vision_outputs.last_hidden_state
        )

        # ----------------------------------------------------
        # Text/gene feature extraction
        # ----------------------------------------------------

        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_hidden = (
            text_outputs.last_hidden_state
        )

        # ----------------------------------------------------
        # Cross-modal decoding
        # ----------------------------------------------------

        logits = self.decoder(
            text_hidden=text_hidden,
            img_tokens=image_tokens
        )

        return logits
```
