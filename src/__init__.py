"""
OmniSight: Research Platform for Robust Semantic Environment Understanding
"""

import torch

# Optimize CPU inference threading on multi-core systems
if torch.get_num_threads() > 2:
    try:
        torch.set_num_threads(2)
    except RuntimeError:
        pass

__version__ = "0.1.0"
