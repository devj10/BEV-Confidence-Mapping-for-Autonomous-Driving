# cs231n
Bird's-Eye-View (BEV) Confidence Mapping for Autonomous Driving

## Setup

### Prerequisites
- Conda (Miniconda or Anaconda)
- NVIDIA GPU with CUDA 11.8+ driver

### Install
```bash
conda env create -f environment.yml
conda activate bev-uncertainty
```

### Verify GPU
```python
import torch
print(torch.cuda.is_available())   # should be True
print(torch.cuda.get_device_name()) # should show your GPU
```

A few practical notes:

- **Change `pytorch-cuda=11.8`** to `12.1` if your driver is newer — run `nvidia-smi` to check your driver version
- **`nuscenes-devkit`** has to be pip-installed, it's not on conda
- **`dropblock` pip package** — worth checking if it works with your PyTorch version; if not, just use the custom `DropBlock2D` class we wrote and drop that line
- If you're on a shared cluster (like a university HPC), they often have CUDA modules you load separately — in that case remove the cudatoolkit line and just match the pytorch-cuda version to what's available

Want me to also write out the full README with the project structure, usage instructions, and how to run the MC inference pipeline?