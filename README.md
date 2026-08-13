# Robust Temporal Echocardiography

Research project on temporal representation learning for
echocardiography video analysis.

## Research Goal

The goal of this project is to investigate robust temporal
representation learning for echocardiography, with a focus on
preserving clinically meaningful cardiac motion while reducing
sensitivity to acquisition and image-quality variations.

## Dataset

Primary dataset:

- EchoNet-Dynamic

The dataset is stored locally and is not included in this repository.

## Baseline

The initial baseline will be based on:

- Vision Transformer (ViT)
- Masked Autoencoder (MAE)
- Temporal representation learning
- Temporal contrastive learning

## Experimental Direction

The project will investigate whether a temporal representation
can remain robust to image-quality and acquisition-related
variations while preserving meaningful cardiac temporal dynamics.

## Project Structure

- `configs/` — Configuration files
- `data/` — Dataset and preprocessing code
- `models/` — Model implementations
- `losses/` — Loss functions
- `training/` — Training scripts
- `experiments/` — Experimental configurations
- `results/` — Experimental results

## Status

Currently setting up the baseline data pipeline and
experimental framework.