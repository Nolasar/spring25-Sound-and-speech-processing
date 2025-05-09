import re
import torch
from typing import List
from g2p_en import G2p

class TextTransform:
    def __init__(self):
        self.g2p = G2p()
        self.phoneme2id = {}
        self.id2phoneme = {}
        self.special_tokens = ['<pad>', '<eos>']
        self.vocab_size = None
        self.max_length = 0

    def fit(self, text_list: List[str]):
        phoneme_set = set()
        for sentence in text_list:
            sentence = self.lower(sentence)
            sentence = self.remove_punct(sentence)
            phonemes = self.text2phonemes(sentence)
            phoneme_set.update(phonemes)

            if len(phonemes) > self.max_length:
                self.max_length = len(phonemes)

        phoneme_array = self.special_tokens + sorted(phoneme_set)
        self.phoneme2id = {p: i for i, p in enumerate(phoneme_array)}
        self.id2phoneme = {i: p for p, i in self.phoneme2id.items()}
        self.vocab_size = len(phoneme_array)
        
    def phonemes2ids(self, phomenes: List[str]):
        if self.vocab_size is None:
            raise RuntimeError("vocab_size is not set. Please call `fit` function to initialize vocabulary before calling phonemes2ids.")
        ids = [self.phoneme2id[phoneme] for phoneme in phomenes]
        return ids + [self.phoneme2id[self.special_tokens[1]]]
    
    def text2phonemes(self, sentence: str):
        phonemes = self.g2p(sentence)
        return phonemes

    def _pad_sequence(self, ids: List[int]) -> List[int]:
        pad_id = self.phoneme2id[self.special_tokens[0]]
        return ids + [pad_id] * (self.max_length + 1 - len(ids))  # +1 to account for <eos>
    
    def __call__(self, sentences:str, padding:bool = True):
        sentence = self.lower(sentences)
        sentence = self.remove_punct(sentence)
        phomenes = self.text2phonemes(sentence)
        ids = self.phonemes2ids(phomenes)
        
        if padding:
            ids = self._pad_sequence(ids)
        
        return torch.tensor(ids, dtype=torch.int).unsqueeze(0)
    
    @staticmethod
    def lower(sentence: str):
        return sentence.lower()
    
    @staticmethod
    def remove_punct(sentence: str):
        return re.sub(r"[^\w\s]", "", sentence).strip()