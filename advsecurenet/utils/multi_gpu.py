import os
import torch.multiprocessing as mp
from torch.utils.data.distibuted import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distibuted import init_process_group, destroy_process_group


def ddp_setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'  # since this is only on one machine
    os.environ['MASTER_PORT'] = '12355'

    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def setup_model(model: torch.nn.Module, rank: int, world_size: int) -> torch.nn.Module:
    ddp_setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    return model
