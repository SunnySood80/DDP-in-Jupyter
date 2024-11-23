
# Distributed Data Parallel (DDP) Setup in Jupyter Notebooks

[Best Practices and Recommendations](#best-practices-and-recommendations)

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup Guide](#setup-guide)
    - [1. Import Necessary Libraries](#1-import-necessary-libraries)
    - [2. Define Helper Functions](#2-define-helper-functions)
        - [a. Finding a Free Port](#a-finding-a-free-port)
        - [b. Setup and Cleanup Functions](#b-setup-and-cleanup-functions)
    - [3. Create Custom Dataset](#3-create-custom-dataset)
    - [4. Define Training Function](#4-define-training-function)
    - [5. Initialize and Run DDP Training](#5-initialize-and-run-ddp-training)


## Best Practices and Recommendations

- **Encapsulate DDP Logic Within Functions**: Prevent unintended recursive executions in Jupyter Notebooks by containing all DDP-related code within functions.
- **Use `torch.multiprocessing` for Process Spawning**: Ensure efficient GPU utilization and isolation.
- **Manage Data Loading with `DistributedSampler`**: Avoid data overlap by assigning unique subsets to each process.
- **Restrict Visualization to Rank 0**: Prevent redundant outputs across GPUs.
- **Dynamic Port Allocation**: Prevent port conflicts during process initialization.

---

## Introduction

Distributed Data Parallel (DDP) is a highly efficient feature in PyTorch that allows for parallel training of models across multiple GPUs. While DDP is commonly used in standalone Python scripts, it can also be effectively implemented within Jupyter Notebooks. This guide provides a step-by-step approach to setting up DDP in a Jupyter Notebook environment, facilitating scalable and high-performance model training.

---

## Prerequisites

Before proceeding, ensure you have the following:

- **Hardware**: A multi-GPU system.
- **Software**:
  - Python 3.x
  - PyTorch with CUDA support
  - Jupyter Notebook
  - Necessary Python libraries: `torch`, `torch.distributed`, `torch.multiprocessing`, `torch.nn.parallel`, `torch.utils.data`, `numpy`, `matplotlib`, `tqdm`, etc.
- **Environment**: Jupyter Notebook running in an environment where all required libraries are installed and accessible.

---

## Setup Guide

### 1. Import Necessary Libraries

Begin by importing all required libraries, including those for DDP, data handling, model training, and visualization.

```python
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
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

#### b. Setup and Cleanup Functions

Initialize and destroy the distributed process group.

```python
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized.")

def cleanup():
    dist.destroy_process_group()
```

### 3. Create Custom Dataset

Define a custom dataset class to handle your specific data structure.

```python
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### 4. Define Training Function

Encapsulate the training logic within a function to ensure proper process management in the notebook.

```python
def train_ddp(rank, world_size, epochs, port, data, labels):
    setup(rank, world_size, port)
    torch.manual_seed(0)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = nn.Linear(data.shape[1], 1).to(device)
    model = DDP(model, device_ids=[rank])
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset and dataloader with DistributedSampler
    dataset = CustomDataset(data, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    cleanup()
```

### 5. Initialize and Run DDP Training

Define the main function to set up and initiate the DDP training process.

```python
def run_ddp_training():
    # Example data loading (replace with actual data)
    # Ensure data and labels are torch tensors
    data = torch.randn(1000, 10)  # 1000 samples, 10 features
    labels = torch.randint(0, 2, (1000,)).float()  # Binary labels
    
    world_size = torch.cuda.device_count()
    epochs = 10
    port = find_free_port()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_ddp, args=(rank, world_size, epochs, port, data, labels))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("DDP training completed.")
```

### Run the training

```python
run_ddp_training()
```

---


- **Monitor GPU Utilization**: Use tools like `nvidia-smi` to ensure balanced workload distribution.
- **Implement Early Stopping and Checkpointing**: Save resources and optimize training time.

---
