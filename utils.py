```python
# utils.py

import json
import os

import torch


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be saved.

    optimizer : torch.optim.Optimizer
        Optimizer whose state will be saved.

    epoch : int
        Current training epoch.

    path : str
        Path used to save the checkpoint.
    """

    # Create the output directory if necessary
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    map_location="cpu"
):
    """
    Load a training checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    map_location : str or torch.device, optional
        Device on which the checkpoint will be loaded.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location
    )

    return checkpoint


def load_json(path):
    """
    Load data from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return data
```
