import copy

import torch
import torch.nn as nn
import numpy as np
try:
    from .model_mamba import FeedForward, BidirectionalMambaLayer, ContinuousBackAndForthMambaBlock
except ImportError:
    from model_mamba import FeedForward, BidirectionalMambaLayer, ContinuousBackAndForthMambaBlock
from typing import Optional
from einops import rearrange


class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


# MambaEncoderCMT
class MambaEncoderCMT(nn.Module):
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


# MambaDecoder
class MambaMultimodalDecoder(nn.Module):
    """
    A non-autoregressive Mamba-based decoder to predict K future trajectories.
    It uses a shared Mamba body and a shared MLP head.
    """
    def __init__(self, d_model: int, num_decoder_layers: int, dropout: float, num_modes: int):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes # This is K

        # The core of the decoder is another MambaEncoder
        decoder_layer = BidirectionalMambaLayer(d_model=d_model, dropout=dropout)
        final_norm = nn.LayerNorm(d_model)
        self.decoder = MambaEncoderCMT(
            encoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=final_norm
        )

        # A single linear head to project the final hidden states to K * (x, y) coordinates
        self.output_head = nn.Linear(d_model, self.num_modes * 2)

    def forward(self, encoder_output: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoder_output (torch.Tensor): The full output from the main encoder,
                                           shape (L, B, D) -> (all_F, B*N, nhid).
            src_key_padding_mask (torch.Tensor, optional): The padding mask, shape (B*N, all_F).

        Returns:
            torch.Tensor: The K predicted future trajectories, shape (pred_len, B*N, K, 2).
        """
        # 1. Process the full sequence again with the decoder layers
        decoder_output = self.decoder(encoder_output, src_key_padding_mask=src_key_padding_mask)

        # 2. Isolate the hidden states corresponding to the future time steps
        future_hidden_states = decoder_output # Shape: (pred_len, B*N, D)

        # 3. Project the future hidden states to K * (x, y) coordinates
        predictions = self.output_head(future_hidden_states) # Shape: (pred_len, B*N, K * 2)

        # 4. Reshape to separate the K modes
        # (pred_len, B*N, K * 2) -> (pred_len, B*N, K, 2)
        pred_traj = rearrange(predictions, 't b (k c) -> t b k c', k=self.num_modes, c=2)

        return pred_traj


# MambaEncoderST
class MambaEncoderST(nn.Module):
    """
    A Mamba-based encoder that is a drop-in replacement for nn.TransformerEncoder.
    It correctly handles the (SeqLen, Batch, Dim) input format.
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
            mask (torch.Tensor, optional): The sequence mask, not used by Mamba.
            src_key_padding_mask (torch.Tensor, optional): The padding mask, shape (B, L).
        """
        # 1. Permute to Batch-First for Mamba: (L, B, D) -> (B, L, D)
        output = src.permute(1, 0, 2)

        # 2. Process through layers
        for mod in self.layers:
            output = mod(output, src_key_padding_mask=src_key_padding_mask)

        # 3. Apply final normalization if provided
        if self.norm is not None:
            output = self.norm(output)

        # 4. Permute back to Sequence-First to match Transformer output format
        output = output.permute(1, 0, 2)

        return output


class TripleGatedFusionModule(nn.Module):
    """
    Fuses three feature tensors (e.g., temporal, spatial, and ego)
    using a learned gating mechanism.
    """
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): The feature dimension (nhid).
        """
        super().__init__()
        # The input to the gate generator is the concatenation of the three feature streams.
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3) # Outputs 3 raw scores for the three weights
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor, ego_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features (torch.Tensor): The first feature stream, shape (T, B*N, D).
            spatial_features (torch.Tensor): The second feature stream, shape (T, B*N, D).
            ego_features (torch.Tensor): The third feature stream, shape (T, B*N, D).

        Returns:
            torch.Tensor: The fused output tensor, shape (T, B*N, D).
        """
        # 1. Concatenate the three features along the last dimension
        gate_input = torch.cat([temporal_features, spatial_features, ego_features], dim=-1)

        # 2. Generate the fusion weights (gates)
        # The softmax ensures the weights sum to 1.
        fusion_weights = torch.softmax(self.gate_generator(gate_input), dim=-1) # Shape (T, B*N, 3)

        # 3. Separate the weights for each stream
        w_temporal = fusion_weights[..., 0:1] # Shape (T, B*N, 1)
        w_spatial  = fusion_weights[..., 1:2] # Shape (T, B*N, 1)
        w_ego      = fusion_weights[..., 2:3] # Shape (T, B*N, 1)

        # 4. Perform the weighted sum to fuse the features
        fused_output = (w_temporal * temporal_features) + \
                       (w_spatial * spatial_features) + \
                       (w_ego * ego_features)

        # 5. Apply final normalization
        return self.output_norm(fused_output)


# Fusion temporal and spatial features using a gating mechanism
class GatedFusionModule(nn.Module):
    """
    Fuses two feature tensors (temporal and spatial) using a learned gating mechanism.
    This is much faster than using a full Mamba block for fusion.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # A small MLP to generate the fusion weights (gates)
        self.gate_generator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2) # Outputs 2 raw scores for the two weights
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, temporal_features: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features (torch.Tensor): Shape (T, B*N, D).
            spatial_features (torch.Tensor): Shape (T, B*N, D).
        """
        # Concatenate along the feature dimension
        gate_input = torch.cat([temporal_features, spatial_features], dim=-1)

        # Generate the fusion weights
        fusion_weights = torch.softmax(self.gate_generator(gate_input), dim=-1) # Shape (T, B*N, 2)

        w_temporal = fusion_weights[..., 0:1] # Shape (T, B*N, 1)
        w_spatial = fusion_weights[..., 1:2]  # Shape (T, B*N, 1)

        # Perform the weighted sum to fuse the features
        fused_output = (w_temporal * temporal_features) + (w_spatial * spatial_features)

        return self.output_norm(fused_output)


class EgoCentricPairwiseInteractionBlock(nn.Module):
    """
    A novel interaction block where the ego agent's current state is processed
    sequentially with the full history of every other agent (including itself).
    The results of these N pairwise interactions are then fused to update the ego's state.
    """
    def __init__(self, d_model: int, num_layers: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        # This Mamba encoder will process the N different interaction sequences
        self.pairwise_mamba_encoder = nn.Sequential(
            *[BidirectionalMambaLayer(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)]
        )

        # An attention mechanism to intelligently fuse the N interaction results
        self.fusion_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features from a temporal encoder, of shape (B, T, N, D).
        Returns:
            torch.Tensor: An output tensor of the same shape (B, T, N, D), where the ego
                          agent's features at the last time step have been updated.
        """
        B, T, N, D = x.shape

        # --- 1. Isolate Ego's Final State and All Agents' Histories ---
        ego_final_state = x[:, -1, 0, :].unsqueeze(1).unsqueeze(1)
        ego_final_state_expanded = ego_final_state.expand(-1, -1, N, -1)

        # --- 2. Construct the N Pairwise Interaction Sequences ---
        interaction_sequences = torch.cat([x, ego_final_state_expanded], dim=1)

        # --- 3. Reshape and Process with Mamba ---
        sequences_reshaped = rearrange(interaction_sequences, 'b t n d -> (b n) t d')
        processed_sequences = self.pairwise_mamba_encoder(sequences_reshaped)

        # --- 4. Extract and Aggregate Fused Tokens ---
        fused_tokens = processed_sequences[:, -1, :]
        interaction_results = rearrange(fused_tokens, '(b n) d -> b n d', b=B, n=N)

        # --- 5. Fuse Interaction Results using Attention ---
        ego_result_query = interaction_results[:, 0:1, :]
        fused_context, _ = self.fusion_attention(
            query=ego_result_query,
            key=interaction_results,
            value=interaction_results
        )
        final_ego_context = self.fusion_norm(ego_result_query + fused_context)

        # --- 6. Create Output Tensor ---
        # Create a copy of the input to avoid in-place modification
        output = x.clone()

        # Update the ego agent's features at the last time step with the new fused context
        output[:, -1, 0, :] = final_ego_context.squeeze(1)

        return output


class EgoCentricPairwiseInteractionBlock_UMamba(nn.Module):
    """
    A novel interaction block where the ego agent's current state is processed
    sequentially with the full history of every other agent (including itself).
    The results of these N pairwise interactions are then fused to update the ego's state.
    """
    def __init__(self, d_model: int, num_layers: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        # This Mamba encoder will process the N different interaction sequences
        self.pairwise_mamba_encoder = nn.Sequential(
            *[ContinuousBackAndForthMambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)]
        )

        # An attention mechanism to intelligently fuse the N interaction results
        self.fusion_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features from a temporal encoder, of shape (B, T, N, D).
        Returns:
            torch.Tensor: An output tensor of the same shape (B, T, N, D), where the ego
                          agent's features at the last time step have been updated.
        """
        B, T, N, D = x.shape

        # --- 1. Isolate Ego's Final State and All Agents' Histories ---
        ego_final_state = x[:, -1, 0, :].unsqueeze(1).unsqueeze(1)
        ego_final_state_expanded = ego_final_state.expand(-1, -1, N, -1)

        # --- 2. Construct the N Pairwise Interaction Sequences ---
        interaction_sequences = torch.cat([x, ego_final_state_expanded], dim=1)

        # --- 3. Reshape and Process with Mamba ---
        sequences_reshaped = rearrange(interaction_sequences, 'b t n d -> (b n) t d')
        processed_sequences = self.pairwise_mamba_encoder(sequences_reshaped)

        # --- 4. Extract and Aggregate Fused Tokens ---
        fused_tokens = processed_sequences[:, -1, :]
        interaction_results = rearrange(fused_tokens, '(b n) d -> b n d', b=B, n=N)

        # --- 5. Fuse Interaction Results using Attention ---
        ego_result_query = interaction_results[:, 0:1, :]
        fused_context, _ = self.fusion_attention(
            query=ego_result_query,
            key=interaction_results,
            value=interaction_results
        )
        final_ego_context = self.fusion_norm(ego_result_query + fused_context)

        # --- 6. Create Output Tensor ---
        # Create a copy of the input to avoid in-place modification
        output = x.clone()

        # Update the ego agent's features at the last time step with the new fused context
        output[:, -1, 0, :] = final_ego_context.squeeze(1)

        return output



# --- The Redesigned Ego-Centric Interaction Block ---
class EgoCentricPairwiseInteractionBlock_UMamba2(nn.Module):
    """
    A novel interaction block where each agent's full trajectory (obs+pred)
    is processed with the ego agent's CURRENT state inserted between the
    observation and prediction phases.
    """
    def __init__(self, d_model: int, num_layers: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.pairwise_mamba_encoder = nn.Sequential(
            *[ContinuousBackAndForthMambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)]
        )
        self.fusion_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, obs_len: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features from a temporal encoder, of shape (B, T, N, D).
            obs_len (int): The length of the observation period.
        Returns:
            torch.Tensor: An output tensor of the same shape (B, T, N, D), where the ego
                          agent's features at the last time step have been updated.
        """
        B, T, N, D = x.shape

        # --- 1. Isolate Ego's CURRENT State and Trajectory Parts ---
        # The "current" state is the ego's state at the last observation step.
        ego_current_state = x[:, obs_len - 1, 0, :].unsqueeze(1).unsqueeze(1) # Shape: (B, 1, 1, D)

        # Split the full trajectory into observation and prediction parts
        obs_part = x[:, :obs_len, :, :]   # Shape: (B, obs_len, N, D)
        pred_part = x[:, obs_len:, :, :]  # Shape: (B, T - obs_len, N, D)

        # --- 2. Construct the N Pairwise Interaction Sequences ---
        # Expand the ego's current state to be inserted into each agent's timeline
        ego_current_state_expanded = ego_current_state.expand(-1, -1, N, -1) # Shape: (B, 1, N, D)

        # Concatenate along the time dimension (T)
        interaction_sequences = torch.cat([obs_part, ego_current_state_expanded, pred_part], dim=1) # Shape: (B, T+1, N, D)

        # --- 3. Reshape and Process with Mamba ---
        sequences_reshaped = rearrange(interaction_sequences, 'b t n d -> (b n) t d')
        processed_sequences = self.pairwise_mamba_encoder(sequences_reshaped)

        # --- 4. Extract and Aggregate Fused Tokens ---
        # The final token of each sequence contains the fused information
        fused_tokens = processed_sequences[:, -1, :]
        interaction_results = rearrange(fused_tokens, '(b n) d -> b n d', b=B, n=N)

        # --- 5. Fuse Interaction Results using Attention ---
        ego_result_query = interaction_results[:, 0:1, :]
        fused_context, _ = self.fusion_attention(
            query=ego_result_query,
            key=interaction_results,
            value=interaction_results
        )
        final_ego_context = self.fusion_norm(ego_result_query + fused_context)

        # --- 6. Create Output Tensor ---
        output = x.clone()
        # Update the ego agent's features at the LAST time step with the new fused context
        output[:, -1, 0, :] = final_ego_context.squeeze(1)

        return output




class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model//2, max_norm=True).to(device)
        self.learned_encoding_pred = nn.Embedding(max_pred_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F, num_people=1) -> torch.Tensor:

        half = x.size(3)//2
        # Bi-directional encoding
        x[:,:in_F,:,0:half*2:2] = x[:,:in_F,:,0:half*2:2] + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,in_F:,:,0:half*2:2] = x[:,in_F:,:,0:half*2:2] + self.learned_encoding_pred(torch.arange(out_F).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(x.size(1), dim=0).to(self.device)).unsqueeze(0)

        return self.dropout(x)


class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)


class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, max_obs_len = 200, max_pred_len = 300, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)


class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000, max_obs_len = 8000, max_pred_len = 12000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)
        self.learned_encoding_pred = nn.Embedding(max_pred_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        ## Bi-directional encoding
        x[:,:in_F] = x[:,:in_F] + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,in_F:] = x[:,in_F:] + self.learned_encoding_pred(torch.arange(out_F).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)


class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000, max_obs_len = 8000, max_pred_len = 12000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding_obs = nn.Embedding(max_obs_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, in_F, out_F) -> torch.Tensor:

        x = x + self.learned_encoding_obs(torch.arange(in_F-1, -1, -1).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)



class TransMotion_UMamba(nn.Module):
    def __init__(self, tok_dim=540, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, nlayers_ego=4, nlayers_goal=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0'):

        super(TransMotion_UMamba, self).__init__()
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.device = device
        self.joints_pose = 39
        self.obs_and_pred = obs_and_pred
        self.obs_and_pred_pose = 15
        self.max_fps = 50


        self.fc_in_traj = nn.Linear(2,nhid)


        self.predict_head_traj = MambaMultimodalDecoder(
                                d_model=nhid,
                                num_decoder_layers=1,
                                dropout=dropout,
                                num_modes=20
                            )


        self.fc_out_pose_3d = nn.Linear(nhid, 3)

        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, device=device)


        mamba_layer = ContinuousBackAndForthMambaBlock(d_model=nhid, d_state=16, d_conv=4, expand=2, dropout=dropout)
        final_norm = nn.LayerNorm(nhid)
        self.local_former = MambaEncoderCMT(encoder_layer=mamba_layer, num_layers=nlayers_local, norm=final_norm)

        self.goal_centric_interaction = EgoCentricPairwiseInteractionBlock(d_model=nhid, num_layers=nlayers_goal, d_state=16, d_conv=4, expand=2, dropout=dropout)
        self.ego_centric_interaction = EgoCentricPairwiseInteractionBlock_UMamba2(d_model=nhid, num_layers=nlayers_ego, d_state=16, d_conv=4, expand=2, dropout=dropout)

        self.gate_fusion = TripleGatedFusionModule(d_model=nhid)

        mamba_layer_global = ContinuousBackAndForthMambaBlock(d_model=nhid, d_state=16, d_conv=4, expand=2, dropout=dropout)
        final_norm_global = nn.LayerNorm(nhid)
        self.global_former = MambaEncoderST(encoder_layer=mamba_layer_global, num_layers=nlayers_global, norm=final_norm_global)


    def forward(self, tgt, padding_mask,metamask=None):

        B, in_F, NJ, K = tgt.shape

        all_F = self.obs_and_pred
        J = 1

        out_F = all_F - in_F
        N = NJ // J

        all_F_pose = self.obs_and_pred_pose
        out_F_pose = all_F_pose - in_F

        # real fps here
        fps = 5

        sampling_stride = int(self.max_fps/fps)

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]
        tgt = tgt.reshape(B,all_F,N,J,K)
        ## add mask
        mask_ratio_traj = 0.0 # 0.1 for pre-training,

        if not metamask:
            mask_ratio_traj = 0.0


        # ## Augment Traj by masking
        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        traj_mask = torch.rand((B,all_F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        tgt_traj = self.fc_in_traj(tgt_traj)
        ## Up-sampling padding
        tgt_traj_temp = tgt_traj.repeat_interleave(sampling_stride, dim=1)
        tgt_traj = self.double_id_encoder(tgt_traj_temp, in_F*sampling_stride, out_F*sampling_stride, num_people=N).reshape(B,all_F,sampling_stride,N,self.nhid)[:,:,0] #[B,all_F,N,128]

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(all_F,-1,self.nhid) # [all_F, B*N, nhid]

        tgt = tgt_traj
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(tgt.size(0),dim=1)

        # [all_F, B*N, nhid]
        # Temporal
        out_local = self.local_former(tgt)


        # goal-centric pairwise interaction block
        tgt_goal_centric = tgt.reshape((all_F),B,N,self.nhid).permute(1,0,2,3) # [B, all_F, N, nhid]
        out_local_goal = self.goal_centric_interaction(tgt_goal_centric) # [B, all_F, N, nhid]
        # [B, all_F, N, nhid] => [all_F, B*N, nhid]
        out_local_goal = out_local_goal.permute(1,0,2,3).reshape(all_F,-1,self.nhid)


        tgt_ego_centric = tgt.reshape((all_F),B,N,self.nhid).permute(1,0,2,3) # [B, all_F, N, nhid]
        out_local_ego = self.ego_centric_interaction(tgt_ego_centric, in_F) # [B, all_F, N, nhid]
        # [B, all_F, N, nhid] => [all_F, B*N, nhid]
        out_local_ego = out_local_ego.permute(1,0,2,3).reshape(all_F,-1,self.nhid)


        # Fuse the local and spatial features
        out_local = self.gate_fusion(out_local, out_local_goal, out_local_ego)
        out_local = out_local * self.output_scale + tgt

        out_traj_pose3d = out_local[:all_F] # [all_F, B*N, nhid]

        out_traj_pose3d = out_traj_pose3d.reshape((all_F),B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        tgt_padding_mask_global = padding_mask.repeat_interleave(all_F, dim=1)
        # out_traj_pose3d: [all_F*N, B, nhid]

        out_global = self.global_former(out_traj_pose3d)
        ##### global residual ######
        out_global = out_global * self.output_scale + out_traj_pose3d
        out_primary = out_global.reshape(N,all_F,B,self.nhid)[0]

        # out_primary[:all_F]: all_F, B, nhid
        out_traj = self.predict_head_traj(out_primary) # [pred_len, B*N, K, 2]
        # all_F, B, n_modes, 2 => n_modes, B, all_F, 2
        out_traj = out_traj.permute(2,1,0,3)

        out_pose_3d = None
        joint_mask = None
        available_num_keypoints = None

        return out_traj, out_pose_3d, joint_mask, available_num_keypoints




def create_model_UMamba(config, logger, devices):
    token_num = config["MODEL"]["token_num"]
    n_hid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]

    nlayers_ego = config["MODEL"]["num_layers_ego"]
    nlayers_goal = config["MODEL"]["num_layers_goal"]

    dropout = config["MODEL"]["dropout"]
    if config["MODEL"]["type"] == "transmotion":
        logger.info("Creating Social-Mamba model.")
        model = TransMotion_UMamba(
            nhid=n_hid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            nlayers_global=nlayers_global,
            nlayers_ego=nlayers_ego,
            nlayers_goal=nlayers_goal,
            dropout=dropout,
            output_scale=config["MODEL"]["output_scale"],
            obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            num_tokens=token_num,
            device=devices
        ).float()
    elif config["MODEL"]["type"] == "transmotion_simple_bimamba":
        logger.info("Creating simplified flattened bidirectional Mamba model.")
        # A simple model that flattens time+agents into a single token sequence,
        # processes with a bidirectional Mamba encoder and decodes via a small head.
        class TransMotion_SimpleBiMamba(nn.Module):
            def __init__(self, nhid, nlayers, num_modes, obs_and_pred, dropout, device='cuda:0'):
                super().__init__()
                self.nhid = nhid
                self.obs_and_pred = obs_and_pred
                self.num_modes = num_modes
                self.device = device

                self.fc_in_traj = nn.Linear(2, nhid)

                mamba_layer = BidirectionalMambaLayer(d_model=nhid, dropout=dropout)
                final_norm = nn.LayerNorm(nhid)
                # Use MambaEncoderCMT which accepts (L, B, D)
                self.encoder = MambaEncoderCMT(encoder_layer=mamba_layer, num_layers=nlayers, norm=final_norm)

                # Predict for all timesteps (obs+pred) per person and per mode
                self.pred_head = nn.Linear(nhid, obs_and_pred * num_modes * 2)

            def forward(self, tgt, padding_mask, metamask=None):
                # Expect tgt: (B, in_F, N, C), where C contains xy in the first two channels.
                B, in_F, N, C = tgt.shape
                all_F = self.obs_and_pred
                out_F = all_F - in_F
                coords = tgt[:, :, :, :2].to(self.device)  # (B, in_F, N, 2)

                x = self.fc_in_traj(coords)  # (B, in_F, N, nhid)

                # Flatten time and agents into a single ordered sequence.
                # This deliberately tests whether a simple bidirectional SSM over serialized agents is enough.
                L = in_F * N
                x_seq = x.permute(1, 0, 2, 3).reshape(L, B, self.nhid)

                enc = self.encoder(x_seq)  # (L, B, nhid)
                enc_per = enc.reshape(in_F, B, N, self.nhid)

                # Match the full Social-Mamba training/eval contract: predict only the focal/ego agent.
                ego_context = enc_per[-1, :, 0]  # (B, nhid)
                pred = self.pred_head(ego_context)  # (B, all_F * num_modes * 2)
                pred = pred.view(B, all_F, self.num_modes, 2)  # (B, all_F, K, 2)

                ego_obs = coords[:, :, 0]  # (B, in_F, 2)
                input_part = ego_obs.permute(1, 0, 2).unsqueeze(2).repeat(1, 1, self.num_modes, 1)
                pred_all = pred.permute(1, 0, 2, 3)  # (all_F, B, K, 2)
                combined = torch.cat([input_part, pred_all[in_F:]], dim=0) if out_F > 0 else input_part

                out_traj = combined.permute(2, 1, 0, 3)  # (K, B, all_F, 2)
                out_pose_3d = None
                joint_mask = None
                available_num_keypoints = None

                return out_traj, out_pose_3d, joint_mask, available_num_keypoints

        model = TransMotion_SimpleBiMamba(
            nhid=n_hid,
            nlayers=nlayers_global,
            num_modes=20,
            obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            dropout=dropout,
            device=devices
        ).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    model = model.to(devices)

    return model
