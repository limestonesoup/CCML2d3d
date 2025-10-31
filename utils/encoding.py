import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, input_dims=3, include_input=True, log_sampling=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.input_dims = input_dims
        self.include_input = include_input
        
        self.output_dims = 0
        if self.include_input:
            self.output_dims += self.input_dims
        self.output_dims += self.input_dims * self.num_freqs * 2
        
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)
            
        self.register_buffer('freq_bands', freq_bands * math.pi)

    def forward(self, inputs):
        # inputs shape: [N, input_dims]
        # self.freq_bands shape: [num_freqs]
        
        # --- FIX: Correctly reshape for broadcasting ---
        # Reshape inputs to [N, 1, input_dims] and freqs to [1, num_freqs, 1]
        # The multiplication will then correctly broadcast to [N, num_freqs, input_dims]
        scaled_inputs = inputs.unsqueeze(1) * self.freq_bands.view(1, -1, 1)
        
        # Get sin and cos and flatten the last two dimensions
        encoded = torch.cat([torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1)
        encoded = encoded.flatten(start_dim=1)
        
        if self.include_input:
            return torch.cat([inputs, encoded], dim=-1)
        else:
            return encoded