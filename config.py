import os
import torch

# Postavke za PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Onemogući CUDA ako nije potrebno
if not torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '' 