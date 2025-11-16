#!/usr/bin/env python3
"""
Clear GPU memory on all available GPUs.
Usage: python clear_gpu_memory.py
"""
import torch
import gc

def clear_all_gpu_memory():
    """Clear memory on all available GPUs."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. No GPUs to clear.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Found {num_gpus} GPU(s)")
    
    # Clear Python garbage collection first
    gc.collect()
    
    # Clear memory on each GPU
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Get memory info
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        
        print(f"✅ GPU {i}: Cleared cache")
        print(f"   Allocated: {allocated:.2f} GB / {total:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
    
    print("\n✅ GPU memory cleared on all devices!")

if __name__ == "__main__":
    clear_all_gpu_memory()

