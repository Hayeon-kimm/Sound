# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Common layers for defining score networks.
"""
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module): # Encoder for CoDi
  def __init__(self, encoder_dim, tdim, FLAGS):
    super(Encoder, self).__init__()
    self.encoding_blocks = nn.ModuleList()
    for i in range(len(encoder_dim)):
      if (i+1)==len(encoder_dim): break
      encoding_block = EncodingBlock(encoder_dim[i], encoder_dim[i+1], tdim, FLAGS)
      self.encoding_blocks.append(encoding_block)

  def forward(self, x, t):
    skip_connections = []
    for encoding_block in self.encoding_blocks:
      x, skip_connection = encoding_block(x, t)
      skip_connections.append(skip_connection)
    return skip_connections, x

class EncodingBlock(nn.Module):
  def __init__(self, dim_in, dim_out, tdim, FLAGS):
    super(EncodingBlock, self).__init__()
    self.layer1 = nn.Sequential( 
        nn.Linear(dim_in, dim_out),
        get_act(FLAGS)
    ) 
    self.temb_proj = nn.Sequential(
        nn.Linear(tdim, dim_out),
        get_act(FLAGS)
    )
    self.layer2 = nn.Sequential(
        nn.Linear(dim_out, dim_out),
        get_act(FLAGS)
    )
    
  def forward(self, x, t):
    x = self.layer1(x).clone()
    x += self.temb_proj(t)
    x = self.layer2(x)
    skip_connection = x
    return x, skip_connection