def get_device():
    """
    Returns the device to work on. The order of preference is:
    
    1. CUDA GPU
    2. Apple Silicon MPS
    3. CPU

    Returns:
        DeviceType: The device to work on.

    """
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
