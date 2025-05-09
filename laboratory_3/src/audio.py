import torch
import torchaudio
import torch.nn.functional as F

from encodec import EncodecModel
from typing import List

class AudioTransform:
    def __init__(self, sample_rate:int = 24000, bandwidth:float = 6.0, device:str = 'cpu'):
        self.sr = sample_rate
        self.bandwidth = bandwidth
        self.device = device

        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(bandwidth)
        self.encodec.to(self.device)
        self.encodec.eval()

        self.max_length = None
        self.pad_id = 0

    def encode(self, waveform:torch.Tensor, sr:int) -> torch.Tensor:
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        
        if waveform.dim() != 3:
            raise ValueError(f"waveform shape must be (B, C, T), now {waveform.shape}")
        
        with torch.no_grad():
            encoded = self.encodec.encode(waveform.to(self.device))
            codes = torch.cat([x[0] for x in encoded], dim=-1)

        return codes.detach().cpu()
    
    def decode(self, codes:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            waveform = self.encodec.decode([(codes.to(self.device), None)])

        return waveform[0].detach().cpu()
    
    def fit(self, codes_list:List[torch.Tensor]):
        self.max_length = max(t.shape[-1] for t in codes_list)

    def pad(self, codes:torch.Tensor) -> torch.Tensor:
        if self.max_length is None:
            raise ValueError(f'max_length is {self.max_length}, fit model to set max_length')
        
        curr_len = codes.shape[-1]
        pad_num = self.max_length - curr_len
        codes_pad = F.pad(codes, (0, pad_num), value=self.pad_id)
        return codes_pad