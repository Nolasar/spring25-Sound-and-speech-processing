import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from src.audio import AudioTransform 
from src.text import TextTransform
from src.model import ModelConfig

class LJSpeechDataset(Dataset):
    def __init__(
            self,
            df:pd.DataFrame,
            audio_transform:AudioTransform, 
            text_transform:TextTransform, 
            wavs_path:str = r'data/LJSpeech-1.1/wavs/',
            ):
        super().__init__()
        self.manifest = df
        self.audio_t = audio_transform
        self.text_t = text_transform
        self.wavs_path = wavs_path

    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        item = self.manifest.iloc[idx]
        
        ids = self.text_t(item['normalized_transcription'])
        path = f'{self.wavs_path}{item["id"]}.wav'

        wave, sr = torchaudio.load(path)
        codes = self.audio_t.encode(wave.unsqueeze(0), sr)
        codes = self.audio_t.pad(codes)

        return ids, codes