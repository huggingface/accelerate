import torch
import intel_extension_for_pytorch as ipex

def main():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0
    print(f"Successfully ran on {num_gpus} GPUs")
    
    if torch.xpu.is_available():
        num_xpus = torch.xpu.device_count()
    else:
        num_xpus = 0
    print(f"Successfully ran on {num_xpus} XPUs")
    


if __name__ == "__main__":
    main()
