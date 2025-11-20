# -*- coding: utf-8 -*-
"""
Veritas Transformer v1.3 – The Polygraph
Wprowadza: VoiceTruthExtractor (Multimodalność) i Equilibrium Bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class VoiceTruthExtractor:
    def extract_features(self, audio_sample: np.ndarray) -> np.ndarray:
        # Symulacja: Prawda = niska wariancja, Kłamstwo = wysoka
        if np.random.rand() > 0.5: return np.array([0.1, 0.05, 0.02, 0.15])
        else: return np.array([0.4, 0.3, 0.15, 0.45])

class VeritasTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.voice_proj = nn.Linear(4, d_model)
        self.out = nn.Linear(d_model, 2) # Binary: Truth/Lie
        
    def forward(self, x: torch.Tensor, voice_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x).mean(dim=1) # Average word embeddings
        voice_emb = self.voice_proj(voice_feat)
        
        # Fuzja
        combined = emb + voice_emb
        
        # Equilibrium Check (K==S=)
        # Symulujemy: czy treść (emb) pasuje do emocji (voice)?
        equilibrium = F.cosine_similarity(emb, voice_emb, dim=-1).mean()
        
        return self.out(combined), equilibrium

def truth_loss(logits, label, eq_bias):
    ce = F.cross_entropy(logits, label)
    # Kara za brak równowagi
    eq_penalty = 0.5 * (1 - eq_bias)
    return ce + eq_penalty

if __name__ == "__main__":
    print("Uruchamianie Veritas v1.3 (Polygraph)...")
    model = VeritasTransformer(10)
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    # Symulacja danych
    x = torch.randint(0, 10, (10, 5))
    voice = torch.randn(10, 4)
    y = torch.randint(0, 2, (10,))
    
    print("Training Results:")
    for i in range(10):
        opt.zero_grad()
        logits, eq = model(x, voice)
        loss = truth_loss(logits, y, eq)
        loss.backward()
        opt.step()
        
        probs = F.softmax(logits, dim=-1)
        density = probs.max(dim=-1)[0].mean().item()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        
        print(f"Epoch {i+1}: Loss {loss.item():.4f}, Density {density:.4f}, Eq Bias {eq.item():.4f}")
    
    print(f"Final Accuracy: {acc:.4f}")
