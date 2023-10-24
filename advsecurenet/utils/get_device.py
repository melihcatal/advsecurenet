def get_device():
    # lazy import to avoid circular imports
    import torch
    import os
    from advsecurenet.shared.types import DeviceType
    # NVIDIA GPU Support
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).is_multiprocessor:
            return DeviceType.CUDA_0
        else:
            return DeviceType.CUDA
    # Apple Silicon Support
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return DeviceType.MPS
    # CPU Support
    else:
        return DeviceType.CPU
