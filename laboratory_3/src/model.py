import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int       # размер словаря фонем (с учетом спецтокенов)    

    D_model: int          # размерность эмбеддингов и внутреннего состояния

    K: int                # число кодебуков
    T_text: int           # длина фонемной последовательности
    T_audio: int          # длина аудио-последовательности
    C: int                # размер каждого кодебука

    pad_idx: int = 0      # индекс паддинга для лосса
    # Гиперпараметры обучения
    epochs: int = 1000
    lr: float = 1e-3
    device: str = 'cpu'

    val_split: float = 0.1
    test_split: float = 0.1
    
    scheduler_step: int = 100  # через сколько эпох снижать lr
    scheduler_gamma: float = 0.1  # коэффициент снижения lr


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.D_model,
            padding_idx=config.pad_idx
        )
        self.conv = nn.Conv1d(
            in_channels=config.D_model,
            out_channels=config.D_model,
            kernel_size=3,
            padding=1
        )

    def forward(self, x_ids: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(x_ids)         # [B, T_ph, D_model]
        x = x.permute(0, 2, 1)            # [B, D_model, T_ph]
        x = F.relu(self.conv(x))          # [B, D_model, T_ph]
        x = x.permute(0, 2, 1)            # [B, T_ph, D_model]
        return x


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=config.D_model,
            out_channels=config.K * config.C,
            kernel_size=1
        )
        self.T_audio = config.T_audio
        self.K = config.K
        self.C = config.C

    def forward(self, X_emb: torch.Tensor) -> torch.Tensor:
        B = X_emb.size(0)
        x = X_emb.permute(0, 2, 1)         # [B, D_model, T_text]
        x = self.proj(x)                   # [B, K*C, T_text]
        x = F.interpolate(
            x,
            size=self.T_audio,
            mode='linear',
            align_corners=False
        )                                  # [B, K*C, T_audio]
        x = x.view(B, self.K, self.C, self.T_audio)  # [B, K, C, T_audio]
        logits = x.permute(0, 1, 3, 2)     # [B, K, T_audio, C]
        return logits