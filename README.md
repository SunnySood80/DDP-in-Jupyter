
# Distributed Data Parallel (DDP) Setup in Jupyter Notebooks

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup Guide](#setup-guide)
    - [1. Import Necessary Libraries](#1-import-necessary-libraries)
    - [2. Define Helper Functions](#2-define-helper-functions)
    - [3. Create Custom Dataset](#3-create-custom-dataset)
    - [4. Setup and Cleanup Functions](#4-setup-and-cleanup-functions)
    - [5. Define Training Function](#5-define-training-function)
    - [6. Initialize and Run DDP Training](#6-initialize-and-run-ddp-training)
4. [Best Practices and Recommendations](#best-practices-and-recommendations)
5. [Complete Example Code](#complete-example-code)
6. [Conclusion](#conclusion)
7. [Contact and Support](#contact-and-support)

---

## Introduction

Distributed Data Parallel (DDP) is a powerful feature in PyTorch that enables efficient training of models across multiple GPUs. While DDP is typically implemented in standalone Python scripts, it can also be effectively utilized within Jupyter Notebooks. This guide provides a detailed walkthrough to set up DDP in a Jupyter Notebook environment, ensuring seamless multi-GPU training without deadlocks or bottlenecks.

---

## Prerequisites

Before proceeding, ensure the following:

- **Hardware**: Access to a multi-GPU system.
- **Software**:
  - Python 3.x
  - PyTorch with CUDA support
  - Necessary Python libraries: `torch`, `torch.distributed`, `torch.multiprocessing`, `torch.nn.parallel`, `torch.utils.data`, `numpy`, `matplotlib`, etc.
- **Environment**: Jupyter Notebook installed and running in an environment where all required libraries are accessible.

---

## Setup Guide

### 1. Import Necessary Libraries

Begin by importing all the required libraries, including those for DDP, data handling, model training, and visualization.

```python
import os
import socket
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from transformers import SamModel, SamProcessor
from monai.losses import DiceLoss
```

### 2. Define Helper Functions

#### a. Finding a Free Port

DDP requires a master address and port for process group initialization. Use a helper function to dynamically find an available port.

```python
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
```

#### b. Compute Metrics

Functions to compute evaluation metrics like Dice Score, IoU, and Accuracy.

```python
def compute_dice_score(predicted, ground_truth):
    intersection = torch.sum(predicted * ground_truth)
    total = torch.sum(predicted) + torch.sum(ground_truth)
    dice_score = (2 * intersection + 1e-6) / (total + 1e-6)
    return dice_score

def compute_iou(predicted, ground_truth):
    intersection = torch.logical_and(predicted, ground_truth)
    union = torch.logical_or(predicted, ground_truth)
    union_sum = torch.sum(union)
    if union_sum == 0:
        return torch.tensor(0.0, device=predicted.device)
    iou_score = torch.sum(intersection).float() / union_sum.float()
    return iou_score

def compute_accuracy(predicted, ground_truth):
    if predicted.numel() == 0 or ground_truth.numel() == 0:
        return torch.tensor(0.0, device=predicted.device)
    pred_flat = predicted.view(-1).bool()
    gt_flat = ground_truth.view(-1).bool()
    correct_pixels = (pred_flat == gt_flat).sum().float()
    total_pixels = torch.tensor(pred_flat.numel(), dtype=torch.float32, device=predicted.device)
    if total_pixels == 0:
        return torch.tensor(0.0, device=predicted.device)
    return correct_pixels / total_pixels
```

#### c. Visualization Function

Function to visualize training and validation results.

```python
def visualize_results(images, points, ground_truths, predicted_masks, epoch, batch_idx, is_training=True):
    for idx in range(min(len(images), 5)):
        image = images[idx].cpu().numpy()
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        else:
            image = image.squeeze()

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        if points[idx].numel() > 0:
            plt.scatter(points[idx][:, 0].cpu().numpy(), points[idx][:, 1].cpu().numpy(), color='red', marker='x')
        plt.title(f"{'Training' if is_training else 'Validation'} Image with Points")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truths[idx].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_masks[idx].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        plt.show()
```

### 3. Create Custom Dataset

Define a custom dataset class to handle your specific data structure.

```python
class CustomDataset(Dataset):
    def __init__(self, images, labels, coordinates):
        self.images = images
        self.labels = labels
        self.coordinates = coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        coords = [(pt[2], pt[1]) for pt in self.coordinates if pt[0] == idx]
        return image, label, coords

def collate_fn(batch):
    images, labels, coords = zip(*batch)
    images = [torch.tensor(image, dtype=torch.float32) for image in images]
    labels = [torch.tensor(label, dtype=torch.float32) for label in labels]
    coords = [torch.tensor([(pt[0], pt[1]) for pt in coord], dtype=torch.float32) for coord in coords]
    return images, labels, coords
```

### 4. Setup and Cleanup Functions

Initialize and destroy the distributed process group.

```python
def setup(rank, world_size, port):
    print(f"Rank {rank}: Initializing process group")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized")

def cleanup():
    dist.destroy_process_group()
```
