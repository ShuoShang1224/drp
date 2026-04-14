import torch
import torch.nn as nn
from drp.model.base_model import BaseModel
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from huggingface_hub import PyTorchModelHubMixin


class PointNetEncoder(nn.Module):
    def __init__(self, output_dim, num_output_tokens, dropout=0):
        super().__init__()

        self.SA_module = PointnetSAModule(
            npoint=num_output_tokens,
            radius=0.1,
            nsample=64,
            mlp=[3, 64, 64, 64],
            bn=False,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64, output_dim*2),
            nn.LayerNorm(output_dim*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(output_dim*2, output_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = xyz.clone().transpose(1, 2).contiguous()
        xyz, features = self.SA_module(xyz, features)
        features = features.transpose(1, 2).contiguous()
        return self.fc_layer(features)


class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class IMPACT(BaseModel, PyTorchModelHubMixin):
    def __init__(
        self,
        hidden_dim=256,
        type_dim=4,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        chunk_size=1,
        pcd_encoders_cfg=None,
        state_encoders_cfg=None,
        transformer_cfg=None,
        normalize_state=False,
        normalize_action=False,
        action_std=None,
        action_space="delta",
    ):
        super().__init__(
            normalize_state=normalize_state,
            normalize_action=normalize_action,
            action_std=action_std,
            action_space=action_space,
        )
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim
        assert self.hidden_dim > self.type_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.pcd_encoders_cfg = pcd_encoders_cfg
        self.state_encoders_cfg = state_encoders_cfg
        self.transformer_cfg = transformer_cfg
        
        # Type embeddings and encodersfor different modalities
        self.type_embeddings = nn.ParameterDict()
        self.encoders = nn.ModuleDict()
        for key, cfg in pcd_encoders_cfg.items():
            if cfg["use_pcd"]:
                self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
                self.encoders[key] = self._initialize_pcd_encoder(self.hidden_dim-self.type_dim, cfg)
        for key, cfg in state_encoders_cfg.items():
            if cfg["use_state"]:
                self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
                self.encoders[key] = StateEncoder(cfg["input_dim"], cfg["hidden_dims"], self.hidden_dim-self.type_dim, cfg["dropout"])
        
        # Action tokens
        self.action_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(chunk_size, hidden_dim)))
        
        # Transformer
        if transformer_cfg["type"] == "encoder_decoder":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["encoder_heads"],
                dim_feedforward=transformer_cfg["encoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg["encoder_layers"])
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["decoder_heads"],
                dim_feedforward=transformer_cfg["decoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_cfg["decoder_layers"])
        else:
            raise NotImplementedError(f"Transformer type {transformer_cfg['type']} not implemented")
        
        # Output head
        self.action_head = nn.Linear(hidden_dim, 7)  # 7 DoF actions

    def _save_pretrained(self, save_directory):
        # hf utils
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

    def _initialize_pcd_encoder(self, hidden_dim, encoder_cfg):
        return PointNetEncoder(hidden_dim, encoder_cfg["num_output_tokens"])
    
    def _add_type_embeddings(self, tokens, token_type):
        B = tokens.shape[0]
        type_emb = self.type_embeddings[token_type].expand(B, tokens.shape[1], -1)
        return torch.cat([tokens, type_emb], dim=-1)
    
    def forward(self, obs):
        # Get inputs
        obs = dict(obs)
        obs = self.normalize_state(obs)
        obs["delta_angles"] = obs["goal_angles"] - obs["current_angles"]  # (B, 7)
        B = obs["scene_pcd"].shape[0]
        
        obs_tokens = []
        for key in self.encoders.keys():
            tokens = self.encoders[key](obs[key])
            if len(tokens.shape) == 2:
                tokens = tokens.unsqueeze(1)
            obs_tokens.append(self._add_type_embeddings(tokens, key))
        obs_tokens = torch.cat(obs_tokens, dim=1)  # (B, N, H)
        
        memory = self.encoder(obs_tokens)  # (B, N, H)
        action_tokens = self.action_tokens.expand(B, -1, -1)  # (B, chunk_size, H)
        output = self.decoder(action_tokens, memory)  # (B, chunk_size, H)
        return self.action_head(output)  # (B, chunk_size, 7)
    
    def forward_pass(self, obs, target=None):
        raise RuntimeError("MPTransformer is inference-only; use forward(obs) or get_action(obs).")
    
    @torch.inference_mode()
    def get_action(self, obs):
        self.eval()
        current_angles = obs["current_angles"]
        pred = self.forward(obs)
        actions = self.decode_actions(current_angles, pred)
        return actions
