# layers/LiftingScheme_Structured.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Splitting(nn.Module):

    def __init__(self, channel_first=True):
        super(Splitting, self).__init__()
        self.channel_first = channel_first
        if channel_first:
            self.conv_even = lambda x: x[:, :, ::2]
            self.conv_odd = lambda x: x[:, :, 1::2]
        else:
            # Note: Original code used channel_first=True, so this branch might be unused
            self.conv_even = lambda x: x[:, ::2, :]
            self.conv_odd = lambda x: x[:, 1::2, :]

    def forward(self, x):
        return self.conv_even(x), self.conv_odd(x)

class MultiScaleConvBlock(nn.Module):
    """Convolutional block using multiple dilation rates within P/U."""
    def __init__(self, in_channels, out_channels, kernel_size, dilations=[1, 2, 4], groups=1):
        super(MultiScaleConvBlock, self).__init__()
        self.dilations = dilations
        self.num_paths = len(dilations)
        self.convs = nn.ModuleList()
        for dilation in dilations:
            # Calculate padding for 'same' output size with dilation
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=padding, dilation=dilation, groups=in_channels) # Keep groups=in_channels like simple_lifting
            )
        # Fusion layer (also grouped)
        self.fusion_conv = nn.Conv1d(out_channels * self.num_paths, out_channels, kernel_size=1, groups=in_channels)

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        fused = torch.cat(outputs, dim=1) # Concatenate along channel dim
        fused = self.fusion_conv(fused)
        return fused

class LiftingSchemeStructuredMultiScale(nn.Module):
    """Lifting scheme using MultiScaleConvBlocks within the original P/U structure."""
    def __init__(self, in_channels, input_size, modified=True, splitting=True, k_size=4, dilations=[1, 2, 4]):
        super(LiftingSchemeStructuredMultiScale, self).__init__()
        self.modified = modified # Typically True for standard lifting
        self.splitting = splitting
        self.split = Splitting(channel_first=True)

        # Size after splitting for LayerNorm
        size_after_split = input_size // 2

        # P and U networks using MultiScaleConvBlock
        self.P = nn.Sequential(
            MultiScaleConvBlock(in_channels, in_channels, k_size, dilations=dilations, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, size_after_split])
        )
        self.U = nn.Sequential(
            MultiScaleConvBlock(in_channels, in_channels, k_size, dilations=dilations, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, size_after_split])
        )

    def forward(self, x):
        if self.splitting:
            x_even, x_odd = self.split(x)
        else:
            # If splitting=False, assumes input is (x_even, x_odd) tuple/list
            x_even, x_odd = x

        # Standard modified lifting steps (Predict P, then Update U)
        if self.modified:
            p_out = self.P(x_even)
            d = x_odd - p_out
            u_out = self.U(d)
            c = x_even + u_out
            return c, d
        else:
            # Original non-modified structure (less common, implement if needed)
            # d = x_odd - self.P(x_even)
            # c = x_even + self.U(d)
            # return c,d
            raise NotImplementedError("Non-modified lifting structure not typically used/implemented here.")


class InverseLiftingSchemeStructuredMultiScale(nn.Module):
    """Inverse lifting scheme with its OWN P/U networks matching the structure."""
    def __init__(self, in_channels, input_size, k_size=4, dilations=[1, 2, 4]):
        """
        input_size: The expected length of the input tensors c and d for this level.
        """
        super(InverseLiftingSchemeStructuredMultiScale, self).__init__()

        # Define P_inv and U_inv networks specific to the inverse pass
        # Their structure mirrors the forward P/U, but LayerNorm uses the current input_size
        self.P_inv = nn.Sequential(
            MultiScaleConvBlock(in_channels, in_channels, k_size, dilations=dilations, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, input_size]) # Use the actual input size 'L' of c/d
        )
        self.U_inv = nn.Sequential(
            MultiScaleConvBlock(in_channels, in_channels, k_size, dilations=dilations, groups=in_channels),
            nn.GELU(),
            nn.LayerNorm([in_channels, input_size]) # Use the actual input size 'L' of c/d
        )
        # We assume the 'modified=True' structure for the inverse calculation
        # (Corresponds to standard Predict-Update lifting)

    def forward(self, c, d):
        # Reverse the modified lifting steps using the dedicated inverse networks
        u_out = self.U_inv(d)
        x_even = c - u_out
        p_out = self.P_inv(x_even)
        x_odd = d + p_out

        # Merge even and odd components
        B, C_dim, L = c.size()
        x = torch.zeros((B, C_dim, 2 * L), dtype=c.dtype, device=c.device)
        x[..., ::2] = x_even
        x[..., 1::2] = x_odd
        return x