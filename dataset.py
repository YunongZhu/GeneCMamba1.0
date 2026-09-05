```python id="pvywku"
# dataset.py

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLIP_MEAN = (
    0.48145466,
    0.4578275,
    0.40821073
)

CLIP_STD = (
    0.26862954,
    0.26130258,
    0.27577711
)


class GeneCaptionDataset(Dataset):
    """
    Dataset for paired histopathology image patches and
    gene-expression captions using a CLIP processor.

    Parameters
    ----------
    images_dir : str
        Directory containing image patches.

    captions : dict
        Mapping from image file names to gene-expression captions.

    tokenizer :
        Tokenizer associated with the CLIP text encoder.

    max_len : int, optional
        Maximum token sequence length.

    processor :
        CLIP processor used for joint image and text preprocessing.
    """

    def __init__(
        self,
        images_dir,
        captions,
        tokenizer,
        max_len=64,
        processor=None
    ):
        self.images_dir = images_dir
        self.captions = list(captions.items())
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processor = processor

        if self.processor is None:
            raise ValueError(
                "GeneCaptionDataset requires a CLIP processor."
            )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name, caption = self.captions[idx]

        image_path = os.path.join(
            self.images_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        encoded = self.processor(
            text=caption,
            images=image,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        pixel_values = (
            encoded["pixel_values"].squeeze(0)
        )

        input_ids = (
            encoded["input_ids"].squeeze(0)
        )

        attention_mask = (
            encoded["attention_mask"].squeeze(0)
        )

        return (
            pixel_values,
            input_ids,
            attention_mask,
            image_name
        )


class CaptionDataset(Dataset):
    """
    Dataset for paired histopathology images and gene-expression
    captions.

    Image preprocessing can be performed either by a CLIP processor
    or by the default CLIP-compatible torchvision transformation.

    Parameters
    ----------
    images_dir : str
        Directory containing image patches.

    captions_dict : dict
        Mapping from image file names to gene-expression captions.

    tokenizer :
        Tokenizer used to encode gene-expression captions.

    max_len : int, optional
        Maximum token sequence length.

    processor : optional
        CLIP processor used for image preprocessing.
    """

    def __init__(
        self,
        images_dir,
        captions_dict,
        tokenizer,
        max_len=64,
        processor=None
    ):
        self.images_dir = images_dir
        self.captions = list(
            captions_dict.items()
        )
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.processor = processor

        if self.processor is None:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=CLIP_MEAN,
                    std=CLIP_STD
                )
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name, caption = self.captions[idx]

        image_path = os.path.join(
            self.images_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        if self.processor is None:
            image_tensor = self.transform(
                image
            )
        else:
            processed = self.processor(
                images=image,
                return_tensors="pt"
            )

            image_tensor = (
                processed["pixel_values"].squeeze(0)
            )

        # ----------------------------------------------------
        # Text/gene caption tokenization
        # ----------------------------------------------------

        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = (
            tokenized["input_ids"].squeeze(0)
        )

        attention_mask = (
            tokenized["attention_mask"].squeeze(0)
        )

        return (
            image_tensor,
            input_ids,
            attention_mask,
            image_name
        )


def collate_fn_with_names(batch):
    """
    Collate dataset samples while retaining image file names.

    Parameters
    ----------
    batch : list
        List of tuples containing:
        (image_tensor, input_ids, attention_mask, image_name)

    Returns
    -------
    tuple
        Batched images, token IDs, attention masks,
        and image file names.
    """

    images = torch.stack(
        [item[0] for item in batch],
        dim=0
    )

    input_ids = torch.stack(
        [item[1] for item in batch],
        dim=0
    )

    attention_mask = torch.stack(
        [item[2] for item in batch],
        dim=0
    )

    image_names = [
        item[3]
        for item in batch
    ]

    return (
        images,
        input_ids,
        attention_mask,
        image_names
    )


def make_dataloader(
    images_dir,
    captions,
    tokenizer,
    batch_size,
    max_len,
    processor=None,
    shuffle=True,
    num_workers=4
):
    """
    Construct a DataLoader for image-caption pairs.

    Parameters
    ----------
    images_dir : str
        Directory containing image patches.

    captions : dict
        Mapping from image names to captions.

    tokenizer :
        Tokenizer used to encode captions.

    batch_size : int
        Number of samples per batch.

    max_len : int
        Maximum token sequence length.

    processor : optional
        CLIP processor used for image preprocessing.

    shuffle : bool, optional
        Whether to shuffle samples.

    num_workers : int, optional
        Number of DataLoader worker processes.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader.
    """

    dataset = CaptionDataset(
        images_dir=images_dir,
        captions_dict=captions,
        tokenizer=tokenizer,
        max_len=max_len,
        processor=processor
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_with_names
    )


def collate_fn(batch):
    """
    Collate dataset samples without retaining image file names.
    """

    images = torch.stack(
        [item[0] for item in batch],
        dim=0
    )

    input_ids = torch.stack(
        [item[1] for item in batch],
        dim=0
    )

    attention_mask = torch.stack(
        [item[2] for item in batch],
        dim=0
    )

    return (
        images,
        input_ids,
        attention_mask
    )
```
