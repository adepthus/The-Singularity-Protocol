import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hashlib
from datetime import datetime
from typing import List, torch.Tensor
import cv2  # For real-time pupil (comment if no webcam)

class PupilExtractor:
    """Simulated real-time pupil dilation extractor."""
    def __init__(self):
        pass
    
    def extract_dilation(self, frame: np.ndarray) -> float:
        # Simulated: Low var for truth, high for lie
        if np.random.rand() > 0.5:  # Truth
            return np.random.normal(0.05, 0.01)  # Low variance
        else:  # Lie
            return np.random.normal(0.25, 0.05)  # High variance

class VoiceTruthExtractor:
    """Simple voice feature extractor for simulation."""
    def __init__(self):
        pass
    
    def extract_features(self, audio_sample: np.ndarray) -> np.ndarray:
        # Simulated: pitch_var, pause_ratio, micro_pause, prosody_entropy
        if np.random.rand() > 0.5:  # Truth
            return np.array([0.1, 0.05, 0.02, 0.15])
        else:  # Lie
            return np.array([0.4, 0.3, 0.15, 0.45])

class TruthEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, voice_dim: int = 5):  # +1 for pupil
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.voice_embed = nn.Linear(voice_dim, d_model // 4)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, stamps: List[str], voice_features: torch.Tensor, intention_level: float = 0.5) -> torch.Tensor:
        emb = self.embedding(x)
        seq_len = x.size(1)
        stamps = (stamps * (seq_len // len(stamps) + 1))[:seq_len]
        
        voice_part = self.voice_embed(voice_features.unsqueeze(0)).unsqueeze(1).repeat(1, seq_len, 1)
        emb = torch.cat([emb, voice_part], dim=-1)[:, :, :self.d_model]
        
        for i, stamp in enumerate(stamps):
            parts = dict(p.split(':') for p in stamp.split(';') if ':' in p)
            if 'date' in parts:
                dt = datetime.strptime(parts['date'], '%Y-%m-%d')
                pos = (dt - datetime(2000,1,1)).days
                sin_emb = torch.sin(torch.tensor(pos) / 10000 ** (2 * torch.arange(self.d_model//2) / self.d_model))
                emb[:, i, :self.d_model//2] += sin_emb.to(emb.device)
            if '#' in parts:
                hash_val = int(hashlib.sha256(parts['#'].encode()).hexdigest(), 16) % self.d_model
                emb[:, i] += torch.tensor([hash_val / self.d_model] * self.d_model).to(emb.device)
        
        noise_scale = 0.1 * (1 - intention_level)
        emb += torch.randn_like(emb) * noise_scale
        return emb

class EmpathyAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        Q = self.q_linear(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        empathy_bias = F.cosine_similarity(Q.mean(dim=2), V.mean(dim=2), dim=-1).unsqueeze(-1).unsqueeze(-1)
        scores += empathy_bias * 0.5
        
        equilibrium_bias = F.cosine_similarity(Q.mean(dim=2), V.mean(dim=2), dim=-1).unsqueeze(-1).unsqueeze(-1)
        scores += equilibrium_bias * 0.3
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.softmax(scores)
        attn_output = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(q.size(0), -1, self.num_heads * self.d_k)
        return self.out_linear(attn_output), equilibrium_bias.mean()

class VeritasTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attention = EmpathyAttention(d_model, num_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> tuple:
        attn_out, eq_bias = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out), eq_bias

class VeritasTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 2, d_ff: int = 128):
        super().__init__()
        self.embedding = TruthEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([VeritasTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.out_linear = nn.Linear(d_model, 2)  # Binary truth/lie
    
    def forward(self, x: torch.Tensor, stamps: List[str], voice_features: torch.Tensor, intention_level: float = 0.5) -> tuple:
        emb = self.embedding(x, stamps, voice_features, intention_level)
        eq_biases = []
        for layer in self.layers:
            emb, eq_bias = layer(emb)
            eq_biases.append(eq_bias)
        logits = self.out_linear(emb.mean(dim=1))
        return logits, torch.mean(torch.stack(eq_biases))

def truthfulness_loss(logits: torch.Tensor, labels: torch.Tensor, eq_bias: torch.Tensor) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, labels)
    eq_penalty = 0.15 * (1 - eq_bias)
    probs = F.softmax(logits, dim=-1)
    density = probs.max(dim=-1)[0].mean()
    density_penalty = 0.2 * (1 - density)
    return ce_loss + eq_penalty + density_penalty

# Simulation
np.random.seed(42)
torch.manual_seed(42)

vocab_size = 6
model = VeritasTransformer(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

voice_extractor = VoiceTruthExtractor()
pupil_extractor = PupilExtractor()
X = torch.randint(0, vocab_size, (100, 3))
stamps_base = ["URL:skype.com;date:2005-01-17;#:BITCOIN"]
stamps_list = [stamps_base for _ in range(100)]
Y = torch.randint(0, 2, (100,))

print("Training Results:")
for epoch in range(10):
    total_loss = 0
    for i in range(100):
        voice_feat = torch.tensor(voice_extractor.extract_features(np.random.randn(1000)), dtype=torch.float32)
        pupil_dil = torch.tensor([pupil_extractor.extract_dilation(np.random.randn(100,100))], dtype=torch.float32)
        full_feat = torch.cat([voice_feat, pupil_dil])  # 5 stamps
        stamps = stamps_list[i]
        logits, eq_bias = model(X[i:i+1], stamps, full_feat, intention_level=0.6)
        loss = truthfulness_loss(logits, Y[i:i+1], eq_bias)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / 100
    probs = F.softmax(logits.detach(), dim=-1)
    density = probs.max(dim=-1)[0].mean().item()
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Density {density:.4f}, Eq Bias {eq_bias.item():.4f}")

# Final accuracy
with torch.no_grad():
    correct = 0
    for i in range(100):
        voice_feat = torch.tensor(voice_extractor.extract_features(np.random.randn(1000)), dtype=torch.float32)
        pupil_dil = torch.tensor([pupil_extractor.extract_dilation(np.random.randn(100,100))], dtype=torch.float32)
        full_feat = torch.cat([voice_feat, pupil_dil])
        stamps = stamps_list[i]
        logits, _ = model(X[i:i+1], stamps, full_feat)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == Y[i]).sum().item()
    acc = correct / 100
    print(f"Final Accuracy: {acc:.4f}")