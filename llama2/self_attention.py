import torch
import torch.nn as nn
from model_args import ModelArgs

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

         
