import os
import yaml
import torch
import numpy as np
from easydict import EasyDict
from torch import nn
from PIL import Image


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to float"""
    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)
    return model


# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def world_info_from_env():
    # from openclip
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def get_cali_data(loader, num_samples):
    cali_data = []
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        cali_data.append(images)
        total += images.size(0)
        if total >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


def denormalize_images(images,
                       mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711)):
    mean = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def save_images_to_file(
        images: torch.Tensor,
        save_path: str = ".",
        filename: str = "Generated_Image",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        suffix: str = ".png",
        digits: int = 3):
    os.makedirs(save_path, exist_ok=True)
    denorm_images = denormalize_images(images, mean, std)
    n = denorm_images.size(0)
    for idx in range(n):
        img_np = (denorm_images[idx].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        out_file = os.path.join(save_path, f"{filename}_{idx:0{digits}d}{suffix}")
        Image.fromarray(img_np).save(out_file)
    print(f"Save {n} Images to Path: {save_path}")