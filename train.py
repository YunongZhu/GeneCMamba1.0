```python
# train.py

import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPTokenizer

from config import (
    BATCH_SIZE,
    CAPTIONS_JSON,
    CLIP_MODEL,
    DECODER_DEPTH,
    DEVICE,
    HIDDEN_DIM,
    IMAGES_DIR,
    LR,
    MAX_TEXT_LEN,
    NUM_EPOCHS,
    PRETRAINED_CLIP,
    PRETRAINED_PATH,
    SAVE_DIR,
    SAVE_EVERY,
    SEED,
    WEIGHT_DECAY,
)
from dataset import make_dataloader
from model import GeneCMambaModel
from utils import load_json, save_checkpoint


def set_seed(seed=42):
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed : int, optional
        Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    max_len=None,
    no_repeat_ngram_size=3,
    compute_topk_k=10,
):
    """
    Evaluate GeneCMamba using token-level and gene-level metrics.

    The validation DataLoader is expected to use batch_size=1 because
    autoregressive generation is performed independently for each sample.

    Parameters
    ----------
    model : torch.nn.Module
        GeneCMamba model.

    dataloader : torch.utils.data.DataLoader
        Validation DataLoader.

    tokenizer :
        CLIP tokenizer.

    device : str
        Device used for inference.

    max_len : int, optional
        Maximum autoregressive generation length.

    no_repeat_ngram_size : int, optional
        Size of repeated n-grams to suppress during generation.

    compute_topk_k : int, optional
        Number of predicted genes used for Top-K gene coverage.

    Returns
    -------
    tuple
        Average validation loss, token accuracy, sample predictions,
        and aggregated gene-level metrics.
    """

    if max_len is None:
        max_len = MAX_TEXT_LEN

    model.eval()

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else 0
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id
    )

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    results = []

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_jaccard = 0.0
    sum_topk = 0.0
    num_samples = 0

    with torch.no_grad():

        for (
            images,
            input_ids,
            attention_mask,
            image_names,
        ) in dataloader:

            if images.size(0) != 1:
                raise ValueError(
                    "Evaluation currently requires batch_size=1 "
                    "for autoregressive generation."
                )

            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # ====================================================
            # Teacher-forced loss and token accuracy
            # ====================================================

            logits = model(
                images,
                input_ids,
                attention_mask,
            )

            logits_shift = (
                logits[:, :-1, :].contiguous()
            )

            targets = (
                input_ids[:, 1:].contiguous()
            )

            loss = criterion(
                logits_shift.view(
                    -1,
                    logits_shift.size(-1),
                ),
                targets.view(-1),
            )

            # Keep the original evaluation convention used in
            # the reported experiments for reproducibility.
            total_loss += (
                loss.item() * targets.numel()
            )

            predictions = torch.argmax(
                logits_shift,
                dim=-1,
            )

            # NOTE:
            # Padding positions are intentionally included here
            # to preserve the metric convention used in the
            # original experiments.
            correct_tokens += (
                predictions == targets
            ).sum().item()

            total_tokens += targets.numel()

            # ====================================================
            # Image feature extraction
            # ====================================================

            vision_outputs = (
                model.clip.vision_model(
                    pixel_values=images
                )
            )

            image_tokens = (
                vision_outputs.last_hidden_state
            )

            # ====================================================
            # Autoregressive gene-token generation
            # ====================================================

            bos_token_id = tokenizer.bos_token_id

            if bos_token_id is None:
                raise ValueError(
                    "The tokenizer does not define a BOS token."
                )

            generated = [
                bos_token_id
            ]

            for _ in range(max_len):

                generated_input_ids = torch.tensor(
                    [generated],
                    dtype=torch.long,
                    device=device,
                )

                generated_attention_mask = (
                    torch.ones_like(
                        generated_input_ids
                    )
                )

                text_outputs = (
                    model.clip.text_model(
                        input_ids=generated_input_ids,
                        attention_mask=(
                            generated_attention_mask
                        ),
                    )
                )

                text_hidden = (
                    text_outputs.last_hidden_state
                )

                generation_logits = (
                    model.decoder(
                        text_hidden,
                        image_tokens,
                    )
                )

                next_token_logits = (
                    generation_logits[:, -1, :]
                )

                # Greedy decoding
                next_token = (
                    next_token_logits
                    .argmax(dim=-1)
                    .item()
                )

                # ================================================
                # No-repeat n-gram constraint
                # ================================================

                if (
                    no_repeat_ngram_size > 0
                    and len(generated)
                    >= no_repeat_ngram_size - 1
                ):

                    candidate_ngram = (
                        generated[
                            -(
                                no_repeat_ngram_size
                                - 1
                            ):
                        ]
                        + [next_token]
                    )

                    repeated = False

                    for index in range(
                        len(generated)
                        - no_repeat_ngram_size
                        + 1
                    ):
                        existing_ngram = (
                            generated[
                                index:
                                index
                                + no_repeat_ngram_size
                            ]
                        )

                        if (
                            existing_ngram
                            == candidate_ngram
                        ):
                            repeated = True
                            break

                    if repeated:

                        num_candidates = min(
                            10,
                            next_token_logits.size(-1),
                        )

                        top_candidates = torch.topk(
                            next_token_logits,
                            k=num_candidates,
                            dim=-1,
                        ).indices[0].tolist()

                        existing_ngrams = [
                            generated[
                                index:
                                index
                                + no_repeat_ngram_size
                            ]
                            for index in range(
                                len(generated)
                                - no_repeat_ngram_size
                                + 1
                            )
                        ]

                        for alternative in (
                            top_candidates
                        ):
                            alternative_ngram = (
                                generated[
                                    -(
                                        no_repeat_ngram_size
                                        - 1
                                    ):
                                ]
                                + [alternative]
                            )

                            if (
                                alternative_ngram
                                not in existing_ngrams
                            ):
                                next_token = (
                                    alternative
                                )
                                break

                generated.append(
                    next_token
                )

                if (
                    next_token
                    == tokenizer.eos_token_id
                ):
                    break

            # ====================================================
            # Decode predictions and ground truth
            # ====================================================

            decoded_prediction = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )

            decoded_target = tokenizer.decode(
                targets[0],
                skip_special_tokens=True,
            )

            # ====================================================
            # Gene-level metrics
            # ====================================================
            #
            # Whitespace-based parsing is preserved here because
            # this is the convention used in the original
            # experiments.
            # ====================================================

            predicted_genes = (
                decoded_prediction.split()
            )

            target_genes = (
                decoded_target.split()
            )

            predicted_set = set(
                predicted_genes
            )

            target_set = set(
                target_genes
            )

            true_positive = len(
                predicted_set & target_set
            )

            false_positive = len(
                predicted_set - target_set
            )

            false_negative = len(
                target_set - predicted_set
            )

            precision = (
                true_positive
                / (
                    true_positive
                    + false_positive
                    + 1e-8
                )
            )

            recall = (
                true_positive
                / (
                    true_positive
                    + false_negative
                    + 1e-8
                )
            )

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                    + 1e-8
                )
            )

            jaccard = (
                true_positive
                / (
                    true_positive
                    + false_positive
                    + false_negative
                    + 1e-8
                )
            )

            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1
            sum_jaccard += jaccard

            # ====================================================
            # Top-K gene coverage
            # ====================================================

            topk_predictions = set(
                predicted_genes[
                    :compute_topk_k
                ]
            )

            topk_true_positive = len(
                topk_predictions & target_set
            )

            topk_score = (
                topk_true_positive
                / (
                    len(target_set)
                    + 1e-8
                )
            )

            sum_topk += topk_score
            num_samples += 1

            results.append(
                {
                    "img": image_names[0],
                    "pred": decoded_prediction,
                    "target": decoded_target,
                    "pred_genes": predicted_genes,
                    "target_genes": target_genes,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "jaccard": jaccard,
                    "topk": topk_score,
                }
            )

    if num_samples == 0:
        raise RuntimeError(
            "The validation DataLoader contains no samples."
        )

    average_loss = (
        total_loss / total_tokens
    )

    token_accuracy = (
        correct_tokens / total_tokens
    )

    aggregated_metrics = {
        "gene_precision": (
            sum_precision / num_samples
        ),
        "gene_recall": (
            sum_recall / num_samples
        ),
        "gene_f1": (
            sum_f1 / num_samples
        ),
        "gene_jaccard": (
            sum_jaccard / num_samples
        ),
        "gene_topk": (
            sum_topk / num_samples
        ),
    }

    return (
        average_loss,
        token_accuracy,
        results,
        aggregated_metrics,
    )


def load_special_genes_from_txt(
    file_path="breast2_genes.txt"
):
    """
    Load additional gene names used to extend the CLIP tokenizer.

    Parameters
    ----------
    file_path : str, optional
        Path to a text file containing one gene name per line.

    Returns
    -------
    list
        Unique gene names converted to uppercase.
    """

    if not os.path.exists(file_path):
        print(
            "[Info] Special gene file not found: "
            f"{file_path}. No additional tokens will be added."
        )
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        genes = [
            line.strip().upper()
            for line in file
            if line.strip()
        ]

    # Stable duplicate removal
    genes = list(
        dict.fromkeys(genes)
    )

    print(
        f"Loaded {len(genes)} special genes."
    )

    return genes


def resize_text_embeddings(
    model,
    tokenizer,
    device,
):
    """
    Resize the CLIP text-token embedding matrix after adding
    gene-specific tokens.

    Existing CLIP embedding weights are preserved.
    Newly added rows retain their random initialization.
    """

    old_embedding = (
        model.clip.text_model
        .embeddings
        .token_embedding
    )

    new_embedding = nn.Embedding(
        num_embeddings=len(tokenizer),
        embedding_dim=(
            old_embedding.embedding_dim
        ),
    )

    with torch.no_grad():
        new_embedding.weight[
            :old_embedding.num_embeddings
        ] = old_embedding.weight

    model.clip.text_model.embeddings.token_embedding = (
        new_embedding.to(device)
    )

    print(
        "Resized CLIP token embedding to:",
        len(tokenizer),
    )


def resize_decoder_output(
    model,
    tokenizer,
    device,
):
    """
    Resize the decoder language-model head to match the tokenizer.
    """

    decoder = model.decoder
    old_head = decoder.lm_head

    new_head = nn.Linear(
        in_features=old_head.in_features,
        out_features=len(tokenizer),
        bias=old_head.bias is not None,
    )

    with torch.no_grad():

        new_head.weight[
            :old_head.out_features,
            :
        ] = old_head.weight

        if old_head.bias is not None:
            new_head.bias[
                :old_head.out_features
            ] = old_head.bias

    decoder.lm_head = (
        new_head.to(device)
    )

    print(
        "Resized decoder output layer to:",
        len(tokenizer),
    )


def write_epoch_log(
    log_file,
    epoch,
    validation_loss,
    token_accuracy,
    gene_metrics,
    results,
):
    """
    Append evaluation results for one epoch to the training log.
    """

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            f"EPOCH {epoch}\n"
        )

        file.write(
            "Validation Loss: "
            f"{validation_loss:.4f}\n"
        )

        file.write(
            "Token Accuracy: "
            f"{token_accuracy * 100:.2f}%\n"
        )

        file.write(
            "Gene Precision: "
            f"{gene_metrics['gene_precision']:.4f}\n"
        )

        file.write(
            "Gene Recall   : "
            f"{gene_metrics['gene_recall']:.4f}\n"
        )

        file.write(
            "Gene F1       : "
            f"{gene_metrics['gene_f1']:.4f}\n"
        )

        file.write(
            "Gene Jaccard  : "
            f"{gene_metrics['gene_jaccard']:.4f}\n"
        )

        file.write(
            "Gene Top-K@10 : "
            f"{gene_metrics['gene_topk']:.4f}\n"
        )

        file.write(
            "\nValidation samples:\n"
        )

        for item in results:

            file.write(
                f"{item['img']}\n"
            )

            file.write(
                f"  Pred:   {item['pred']}\n"
            )

            file.write(
                f"  Target: {item['target']}\n\n"
            )

        file.write(
            "=" * 80 + "\n"
        )


def train():
    """
    Train GeneCMamba using paired histopathology images and
    gene-expression captions.
    """

    set_seed(SEED)

    device = DEVICE

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Load dataset annotations
    # ========================================================

    captions = load_json(
        CAPTIONS_JSON
    )

    all_items = list(
        captions.items()
    )

    random.shuffle(
        all_items
    )

    num_train = int(
        len(all_items) * 0.9
    )

    train_items = (
        all_items[:num_train]
    )

    validation_items = (
        all_items[num_train:]
    )

    train_captions = dict(
        train_items
    )

    validation_captions = dict(
        validation_items
    )

    print(
        "\nDataset split:"
    )

    print(
        f"  Training samples   : "
        f"{len(train_captions)}"
    )

    print(
        f"  Validation samples : "
        f"{len(validation_captions)}"
    )

    # ========================================================
    # Tokenizer and processor
    # ========================================================
    #
    # CLIP_MODEL can be either:
    #   - a Hugging Face model identifier
    #   - a local directory containing CLIP files
    #
    # Example:
    #   "openai/clip-vit-base-patch32"
    #   "./Clip-vit-base-patch32"
    # ========================================================

    tokenizer = (
        CLIPTokenizer.from_pretrained(
            CLIP_MODEL
        )
    )

    processor = (
        CLIPProcessor.from_pretrained(
            CLIP_MODEL
        )
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_dataloader = make_dataloader(
        images_dir=IMAGES_DIR,
        captions=train_captions,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_TEXT_LEN,
        processor=processor,
        shuffle=True,
    )

    validation_dataloader = (
        make_dataloader(
            images_dir=IMAGES_DIR,
            captions=validation_captions,
            tokenizer=tokenizer,
            batch_size=1,
            max_len=MAX_TEXT_LEN,
            processor=processor,
            shuffle=False,
        )
    )

    # ========================================================
    # Build GeneCMamba
    # ========================================================

    model = GeneCMambaModel(
        clip_model_name=CLIP_MODEL,
        decoder_dim=HIDDEN_DIM,
        decoder_depth=DECODER_DEPTH,
        vocab_size=tokenizer.vocab_size,
        pretrained_clip=PRETRAINED_CLIP,
        pretrained_path=PRETRAINED_PATH,
    )

    model.to(
        device
    )

    # ========================================================
    # Extend vocabulary with gene-specific tokens
    # ========================================================

    special_genes = (
        load_special_genes_from_txt(
            "breast2_genes.txt"
        )
    )

    num_added = tokenizer.add_tokens(
        special_genes
    )

    print(
        f"Added {num_added} new gene tokens."
    )

    if num_added > 0:

        resize_text_embeddings(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        resize_decoder_output(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    # ========================================================
    # Freeze CLIP and train only the GeneCMamba decoder
    # ========================================================
    #
    # This preserves the optimization strategy used in the
    # original experiments.
    # ========================================================

    for name, parameter in (
        model.named_parameters()
    ):
        parameter.requires_grad = (
            "decoder" in name
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id
            is not None
            else 0
        )
    )

    use_amp = (
        torch.cuda.is_available()
        and str(device).startswith("cuda")
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    os.makedirs(
        SAVE_DIR,
        exist_ok=True,
    )

    # ========================================================
    # Training log
    # ========================================================

    log_file = os.path.join(
        SAVE_DIR,
        "training_log.txt",
    )

    best_accuracy = -1.0
    best_epoch = 0
    best_results = None
    best_gene_metrics = None

    with open(
        log_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "GeneCMamba Training Log\n"
        )

        file.write(
            "=" * 80 + "\n"
        )

        file.write(
            f"Training samples: "
            f"{len(train_captions)}\n"
        )

        file.write(
            f"Validation samples: "
            f"{len(validation_captions)}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: {LR}\n"
        )

        file.write(
            f"Weight decay: "
            f"{WEIGHT_DECAY}\n"
        )

        file.write(
            f"Maximum text length: "
            f"{MAX_TEXT_LEN}\n"
        )

        file.write(
            f"Decoder depth: "
            f"{DECODER_DEPTH}\n"
        )

        file.write(
            f"Decoder dimension: "
            f"{HIDDEN_DIM}\n"
        )

        file.write(
            f"Random seed: {SEED}\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_dataloader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/"
                f"{NUM_EPOCHS}"
            ),
            ncols=120,
        )

        for (
            step,
            (
                images,
                input_ids,
                attention_mask,
                _,
            ),
        ) in enumerate(progress):

            images = images.to(
                device
            )

            input_ids = input_ids.to(
                device
            )

            attention_mask = (
                attention_mask.to(
                    device
                )
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):

                logits = model(
                    images,
                    input_ids,
                    attention_mask,
                )

                logits_shift = (
                    logits[
                        :, :-1, :
                    ].contiguous()
                )

                targets = (
                    input_ids[
                        :, 1:
                    ].contiguous()
                )

                loss = criterion(
                    logits_shift.view(
                        -1,
                        logits_shift.size(-1),
                    ),
                    targets.view(-1),
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
            )

            average_training_loss = (
                running_loss
                / (step + 1)
            )

            progress.set_postfix(
                {
                    "loss": (
                        f"{loss.item():.4f}"
                    ),
                    "avg_loss": (
                        f"{average_training_loss:.4f}"
                    ),
                    "lr": (
                        optimizer
                        .param_groups[0]["lr"]
                    ),
                }
            )

        # ====================================================
        # Validation
        # ====================================================

        (
            validation_loss,
            validation_accuracy,
            validation_results,
            gene_metrics,
        ) = evaluate(
            model=model,
            dataloader=(
                validation_dataloader
            ),
            tokenizer=tokenizer,
            device=device,
            max_len=MAX_TEXT_LEN,
            no_repeat_ngram_size=3,
            compute_topk_k=10,
        )

        print(
            f"\nEpoch {epoch + 1}"
        )

        print(
            "- Token Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        print(
            "- Gene F1: "
            f"{gene_metrics['gene_f1']:.2f} | "
            "Recall: "
            f"{gene_metrics['gene_recall']:.2f} | "
            "Precision: "
            f"{gene_metrics['gene_precision']:.2f}"
        )

        print(
            "- Jaccard: "
            f"{gene_metrics['gene_jaccard']:.2f} | "
            "Top-K@10: "
            f"{gene_metrics['gene_topk']:.2f}"
        )

        print(
            "-" * 50
        )

        write_epoch_log(
            log_file=log_file,
            epoch=epoch + 1,
            validation_loss=(
                validation_loss
            ),
            token_accuracy=(
                validation_accuracy
            ),
            gene_metrics=(
                gene_metrics
            ),
            results=(
                validation_results
            ),
        )

        # ====================================================
        # Track the best model according to Token Accuracy
        # ====================================================

        if (
            validation_accuracy
            > best_accuracy
        ):

            best_accuracy = (
                validation_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            best_results = (
                validation_results
            )

            # Important:
            # Store metrics from the same best epoch instead of
            # accidentally using the final epoch's metrics.
            best_gene_metrics = (
                gene_metrics.copy()
            )

        # ====================================================
        # Save checkpoint
        # ====================================================

        if (
            (epoch + 1)
            % SAVE_EVERY
            == 0
        ):

            checkpoint_path = os.path.join(
                SAVE_DIR,
                (
                    "genecmamba_"
                    f"epoch{epoch + 1}.pt"
                ),
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                path=checkpoint_path,
            )

            print(
                "\nSaved checkpoint to: "
                f"{checkpoint_path}"
            )

    # ========================================================
    # Write best-epoch summary
    # ========================================================

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n"
            "==== BEST TOKEN ACCURACY MODEL ====\n"
        )

        file.write(
            f"Epoch {best_epoch} | "
            "Token Accuracy: "
            f"{best_accuracy * 100:.2f}%\n"
        )

        file.write(
            "Full Gene Metrics:\n"
        )

        file.write(
            "  Precision: "
            f"{best_gene_metrics['gene_precision']:.4f}\n"
        )

        file.write(
            "  Recall   : "
            f"{best_gene_metrics['gene_recall']:.4f}\n"
        )

        file.write(
            "  F1       : "
            f"{best_gene_metrics['gene_f1']:.4f}\n"
        )

        file.write(
            "  Jaccard  : "
            f"{best_gene_metrics['gene_jaccard']:.4f}\n"
        )

        file.write(
            "  Top-K@10 : "
            f"{best_gene_metrics['gene_topk']:.4f}\n\n"
        )

        file.write(
            "Best Validation Samples:\n"
        )

        for item in best_results:

            file.write(
                f"{item['img']}\n"
            )

            file.write(
                f"  Pred:   "
                f"{item['pred']}\n"
            )

            file.write(
                f"  Target: "
                f"{item['target']}\n\n"
            )

        file.write(
            "=" * 80 + "\n"
        )

    print(
        "\nTraining finished."
    )

    print(
        "Best Token Accuracy: "
        f"{best_accuracy * 100:.2f}% "
        f"at Epoch {best_epoch}"
    )


if __name__ == "__main__":
    train()
```
