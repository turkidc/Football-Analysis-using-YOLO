import torch
import os
import asyncio

def init_torch():
    """Inicijalizira PyTorch s optimalnim postavkama."""
    # Postavke za PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Onemogući CUDA ako nije potrebno
    if not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Postavi PyTorch u CPU mod koristeći nove preporučene metode
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    
    # Postavi broj niti na 1
    torch.set_num_threads(1)
    
    # Postavi environment varijable za PyTorch
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    
    # Pokušaj postaviti event loop ako ne postoji
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    return torch.device('cpu') 