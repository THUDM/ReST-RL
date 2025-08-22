import torch
from typing import Union
from rms.reward_models import BaseRM, TransformersScalarRM, TransformersProbRM
from rms.reward_models_extensions import SkyworkRM, InternRM, GeneralizableRM, QwenRM, RLHFlowRM


def deploy_rm(rm_class: str = 'standard_scalar', backend: str = None, rm_type: str = 'prm',
              device: Union[str, torch.device] = 'cuda', max_length: int = 1024) -> BaseRM:
    if rm_class == 'transformers_scalar':
        return TransformersScalarRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'transformers_prob':
        return TransformersProbRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'skywork':
        return SkyworkRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'intern':
        return InternRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'generalizable':
        return GeneralizableRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'qwen':
        return QwenRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    elif rm_class == 'rlhflow':
        return RLHFlowRM(backend=backend, rm_type=rm_type, device=device, max_length=max_length)
    else:
        raise ValueError('Invalid reward model class')
