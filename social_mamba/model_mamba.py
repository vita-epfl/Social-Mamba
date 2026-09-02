import copy

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from typing import Optional
import math

class FeedForward(nn.Module):
    """A simple two-layer MLP with GELU activation, used as the FFN block."""
    def __init__(self, d_model: int, d_ff: int = 0, dropout: float = 0.2):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class ContinuousBackAndForthMambaBlock(nn.Module):
    """
    A novel Mamba layer that processes a sequence concatenated with its reverse
    [L, L_reversed] to learn continuity in a single pass.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # A single Mamba block to process the entire 2L sequence
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # The final FFN to process the output
        self.ffn = FeedForward(d_model=d_model, d_ff=2 * d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, L, D) -> (Batch, Length, Dim)
        """
        B, L, D = x.shape
        residual = x
        x_norm = self.norm(x)

        # --- 1. Create the continuous 2L sequence ---
        x_reversed = torch.flip(x_norm, dims=[1])
        x_continuous = torch.cat([x_norm, x_reversed], dim=1) # Shape: (B, 2*L, D)

        # --- 2. Process the full 2L sequence in one pass ---
        mamba_out_2L = self.mamba(x_continuous)

        # --- 3. Split and Combine the Forward and Backward Outputs ---
        # The first L elements correspond to the forward pass context
        forward_context = mamba_out_2L[:, :L, :]

        # The last L elements correspond to the backward pass context
        backward_context_reversed = mamba_out_2L[:, L:, :]
        # Flip it back to the original order
        backward_context = torch.flip(backward_context_reversed, dims=[1])

        # Combine the context from both directions
        combined_context = forward_context + backward_context

        # --- 4. Apply FFN and the final residual connection ---
        output = residual + self.ffn(combined_context)

        return output


class BidirectionalMambaLayer(nn.Module):
    """
    A single layer that replaces nn.TransformerEncoderLayer.
    It consists of a Bidirectional Mamba block followed by a Feed-Forward block.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        # --- Mamba Sub-block ---
        self.mamba_norm = nn.LayerNorm(d_model)
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # --- FFN Sub-block ---
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model*2, dropout=dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (B, L, D) -> (Batch, Length, Dim)
            src_key_padding_mask (torch.Tensor, optional): Mask of shape (B, L).
                                                           True indicates a padded value.
        """
        # --- 1. Mamba Block with Pre-LN and Residual ---
        mamba_residual = x
        x_norm = self.mamba_norm(x)

        forward_out = self.forward_mamba(x_norm)

        x_reversed = torch.flip(x_norm, dims=[1])
        backward_out = self.backward_mamba(x_reversed)

        x_mamba = mamba_residual + forward_out + torch.flip(backward_out, dims=[1])

        # --- 2. FFN Block with Pre-LN and Residual ---
        ffn_residual = x_mamba
        x_mamba_norm = self.ffn_norm(x_mamba)
        x_ffn = self.ffn(x_mamba_norm)
        output = ffn_residual + x_ffn

        # --- 3. Apply padding mask if provided ---
        if src_key_padding_mask is not None:
            # The mask is (B, L), we need (B, L, 1) to broadcast over the feature dimension.
            # We also invert the mask because PyTorch uses True for padding, but we want to multiply by 0.
            output = output * (~src_key_padding_mask).unsqueeze(-1)

        return output


class ContinuousUNetMambaLayer(nn.Module):
    """
    A Mamba layer that processes a continuous [L, L_reversed] sequence while using
    U-Net-like state fusion between the forward and backward passes.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        # All projections now operate on the 2L sequence length
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = rearrange(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> 1 n").repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Input x has shape (B, L, D)
        """
        B, L, D = x.shape
        residual = x

        # --- 1. Create the continuous 2L sequence ---
        x_continuous = torch.cat([x, torch.flip(x, dims=[1])], dim=1) # Shape: (B, 2*L, D)

        # --- Mamba Input Path on the 2L sequence ---
        xz = self.in_proj(x_continuous)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = self.conv1d(x_ssm)[:, :, :2*L]
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = torch.nn.functional.silu(x_ssm)

        x_dbl = self.x_proj(x_ssm)
        delta, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())

        # --- U-Net State Fusion Scan on the 2L sequence ---
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []

        # The scan now runs over the full 2L sequence
        for i in range(2 * L):
            delta_i = self.dt_proj(delta[:, i, :])
            delta_A_t = torch.exp(delta_i.unsqueeze(-1) * A)

            # The "skip connection" is the input x_ssm itself
            skip_connection_input = x_ssm[:, i, :]

            delta_B_x_t = (delta_i * skip_connection_input).unsqueeze(-1) * B_ssm[:, i, :].unsqueeze(1)

            h = delta_A_t * h + delta_B_x_t

            y_i = (h @ C_ssm[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)

        # --- Final Output Calculation ---
        y = y + x_ssm * self.D
        y = y * torch.nn.functional.silu(z)

        output_2L = self.out_proj(y)

        # Truncate the output to the original length L
        output_L = output_2L[:, :L, :]

        return self.norm(output_L + residual)


class MambaEncoder(nn.Module):
    """
    A Mamba-based encoder that is a drop-in replacement for nn.TransformerEncoder.
    """
    def __init__(self, encoder_layer, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): The input sequence, shape (L, B, D).
            mask (torch.Tensor, optional): The sequence mask, not used by Mamba but kept for API compatibility.
            src_key_padding_mask (torch.Tensor, optional): The padding mask, shape (B, L).
        """
        # --- 1. Permute to Batch-First for Mamba ---
        # (L, B, D) -> (B, L, D)
        output = src.permute(1, 0, 2)

        # --- 2. Process through layers ---
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask)

        # --- 3. Apply final normalization if provided ---
        if self.norm is not None:
            output = self.norm(output)

        # --- 4. Permute back to Sequence-First to match Transformer output ---
        # (B, L, D) -> (L, B, D)
        output = output.permute(1, 0, 2)

        return output


class MambaTrajectoryDecoder(nn.Module):
    """
    A non-autoregressive Mamba-based decoder to predict future trajectories.
    This replaces a bank of MLP heads.
    """
    def __init__(self, d_model: int, num_decoder_layers: int, pred_len: int):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # The core of the decoder is another MambaEncoder
        decoder_layer = BidirectionalMambaLayer(d_model=d_model, d_state=16, d_conv=4, expand=2, dropout=0.2)
        final_norm = nn.LayerNorm(d_model)
        self.decoder = MambaEncoder(
            encoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=final_norm
        )

        # A single linear head to project the final hidden states to (x, y) coordinates
        self.output_head = nn.Linear(d_model, 2)

    def forward(self, encoder_output: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor): The full output from the main encoder,
                                           shape (L, B, D) -> (all_F, B*N, nhid).
            src_key_padding_mask (torch.Tensor, optional): The padding mask, shape (B*N, all_F).

        Returns:
            torch.Tensor: The predicted future trajectory, shape (pred_len, B*N, 2).
        """
        # 1. Process the full sequence again with the decoder layers
        # This allows the model to refine the representations with a focus on prediction.
        decoder_output = self.decoder(encoder_output, src_key_padding_mask=src_key_padding_mask)

        # 2. Isolate the hidden states corresponding to the future time steps
        # The input was (all_F, B*N, D), so we slice the first dimension.
        future_hidden_states = decoder_output

        # 3. Project the future hidden states to (x, y) coordinates
        # The Linear layer is applied to the last dimension (D)
        pred_traj = self.output_head(future_hidden_states)

        return pred_traj
