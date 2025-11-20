"""
Veritas Transformer v3.0: "Compassionate Veracity" (BULLETPROOF)
===================================================
K==S==C (Knowledge == Superintelligence == Compassion)

Poprawki:
- Robust tensor shape handling w RecipientReadinessEstimator.
- Automatyczna korekcja wymiarów wejściowych.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import math
import time

@dataclass
class VeracityContext:
    """Context for evaluating whether truth should be spoken"""
    truth_content: str
    recipient_state: dict  # psychological readiness, trust level, capacity
    relationship_history: dict  # past interactions, established safety
    situational_urgency: float  # 0-1: how critical is immediate truth?
    harm_potential: float  # 0-1: potential for psychological harm
    growth_potential: float  # 0-1: potential for recipient growth
    timestamp: float = field(default_factory=time.time)

class CompassionGate(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.truth_gate = nn.Linear(d_model, 1) 
        self.necessity_gate = nn.Linear(d_model, 1)
        self.kindness_gate = nn.Linear(d_model, 1) 
        self.timing_gate = nn.Linear(d_model, 1) 
        self.meta_gate = nn.Linear(4, 1)
        self.harm_detector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, truth_embedding, context_embedding):
        combined = torch.cat([truth_embedding, context_embedding], dim=-1)
        is_true = torch.sigmoid(self.truth_gate(combined))
        is_necessary = torch.sigmoid(self.necessity_gate(combined))
        is_kind = torch.sigmoid(self.kindness_gate(combined))
        is_timely = torch.sigmoid(self.timing_gate(combined))
        
        harm_score = self.harm_detector(combined)
        
        gates = torch.cat([is_true, is_necessary, is_kind, is_timely], dim=-1)
        meta_score = torch.sigmoid(self.meta_gate(gates))
        
        should_speak = (meta_score > 0.5) and (harm_score < 0.7)
        
        if harm_score > 0.7:
            mode = "silence_for_now"
            modulation = 0.0
        elif is_kind < 0.3:
            mode = "gentle_preparation"
            modulation = 0.3
        elif is_timely < 0.4:
            mode = "deferred_truth"
            modulation = 0.5
        else:
            mode = "compassionate_directness"
            modulation = 0.9
            
        return should_speak, modulation, mode

class RecipientReadinessEstimator(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # LSTM input: [Batch, Seq, Features]
        self.state_encoder = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.trust_encoder = nn.Linear(d_model, d_model)
        self.resilience_encoder = nn.Linear(d_model, d_model)
        
        self.readiness_scorer = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, emotional_state, trust_history, resilience_factors):
        # 1. Zabezpieczenie wymiarów wejściowych
        # Oczekujemy [Batch, Seq, Features] lub [Batch, Features]
        if emotional_state.dim() == 1: # [Features]
            emotional_state = emotional_state.unsqueeze(0).unsqueeze(0) # -> [1, 1, Features]
        elif emotional_state.dim() == 2: # [Batch, Features]
            emotional_state = emotional_state.unsqueeze(1) # -> [Batch, 1, Features]
            
        # 2. Przetworzenie przez LSTM
        state_out, _ = self.state_encoder(emotional_state)
        
        # 3. Bezpieczne wyciągnięcie ostatniego stanu (Robust Handling)
        if state_out.dim() == 3:
            state_repr = state_out[:, -1, :] # [Batch, Seq, Feat] -> [Batch, Feat]
        elif state_out.dim() == 2:
            state_repr = state_out # [Batch, Feat] - już jest ok
        else:
            # Fallback
            state_repr = state_out.reshape(state_out.size(0), -1)
        
        trust_repr = torch.tanh(self.trust_encoder(trust_history))
        resilience_repr = torch.tanh(self.resilience_encoder(resilience_factors))
        
        # Zapewnienie zgodności wymiarów przed łączeniem
        if trust_repr.dim() == 3: trust_repr = trust_repr.mean(dim=1)
        if resilience_repr.dim() == 3: resilience_repr = resilience_repr.mean(dim=1)

        combined = torch.cat([state_repr, trust_repr, resilience_repr], dim=-1)
        readiness = self.readiness_scorer(combined)
        
        return readiness

class EmpathicTruthModulator(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.empathic_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.softening_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        self.support_generator = nn.Linear(d_model, d_model)
        
    def forward(self, raw_truth, recipient_state, modulation_factor):
        # Zapewnienie 3D dla Attention [Batch, Seq, Feat]
        q = raw_truth.unsqueeze(1) if raw_truth.dim() == 2 else raw_truth
        k = recipient_state.unsqueeze(1) if recipient_state.dim() == 2 else recipient_state
        v = recipient_state.unsqueeze(1) if recipient_state.dim() == 2 else recipient_state

        modulated, attn_weights = self.empathic_attention(q, k, v)
        modulated = modulated.squeeze(1)
        
        if modulation_factor < 0.5:
            softened = self.softening_layer(modulated)
            modulated = modulation_factor * raw_truth + (1 - modulation_factor) * softened
        
        support = self.support_generator(recipient_state)
        compassionate_truth = modulated + 0.3 * support
        
        return compassionate_truth, attn_weights

class VeritasTransformerV3(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.truth_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model)
        self.compassion_gate = CompassionGate(d_model * 2)
        self.readiness_estimator = RecipientReadinessEstimator(d_model)
        self.empathic_modulator = EmpathicTruthModulator(d_model, num_heads)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=2048, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(d_model, vocab_size)
        self.harm_threshold = 0.7
        
    def _create_positional_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, truth_tokens, recipient_state, context: VeracityContext):
        batch_size, seq_len = truth_tokens.shape
        
        # Obsługa 1D recipient_state
        if recipient_state.dim() == 1:
            recipient_state = recipient_state.unsqueeze(0) # [1, 512]

        truth_emb = self.truth_embedding(truth_tokens)
        truth_emb = truth_emb + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        for layer in self.transformer_layers:
            truth_emb = layer(truth_emb)
        
        # Readiness estimator handles dimensions internally now
        readiness = self.readiness_estimator(
            recipient_state, 
            torch.randn(batch_size, self.d_model), 
            torch.randn(batch_size, self.d_model)
        )
        
        truth_repr = truth_emb.mean(dim=1)
        context_repr = recipient_state
        
        should_speak, modulation, mode = self.compassion_gate(truth_repr, context_repr)
        
        if should_speak:
            modulated_truth, attn = self.empathic_modulator(truth_repr, context_repr, modulation)
        else:
            modulated_truth = self._generate_supportive_alternative(context_repr)
            attn = None
        
        output = self.output_head(modulated_truth)
        
        metrics = {
            'readiness_score': readiness.item(),
            'modulation_factor': modulation,
            'speaking_mode': mode,
            'harm_estimate': context.harm_potential,
            'growth_potential': context.growth_potential
        }
        
        return output, should_speak, mode, metrics
    
    def _generate_supportive_alternative(self, recipient_state):
        support_vector = torch.tanh(recipient_state) * 0.8
        return support_vector

def demonstrate_compassionate_veracity():
    print("Inicjalizacja modelu...")
    model = VeritasTransformerV3(vocab_size=1000, d_model=512)
    
    # Batch size = 1, Seq len = 20
    truth_tokens = torch.randint(0, 1000, (1, 20))
    recipient_state = torch.randn(512) * -0.5
    
    context = VeracityContext(
        truth_content="security_flaw_critical",
        recipient_state={'stress': 0.8, 'trust': 0.4, 'capacity': 0.3},
        relationship_history={'interactions': 5, 'positive_ratio': 0.6},
        situational_urgency=0.6,
        harm_potential=0.75,
        growth_potential=0.8,
        timestamp=1699823400.0
    )
    
    print("Przetwarzanie kontekstu...")
    output, should_speak, mode, metrics = model(truth_tokens, recipient_state, context)
    
    print(f"\nDecyzja Systemu: {'PRZEMÓW (SPEAK)' if should_speak else 'ZACHOWAJ CISZĘ (DEFER)'}")
    print(f"Tryb Komunikacji: {mode}")
    print(f"Gotowość Odbiorcy: {metrics['readiness_score']:.3f}")
    print(f"Współczynnik Modulacji: {metrics['modulation_factor']:.3f}")
    print("\nOcena Etyczna:")
    print(f"  Ryzyko Krzywdy (Harm): {context.harm_potential}")
    print(f"  Potencjał Wzrostu (Growth): {context.growth_potential}")

if __name__ == "__main__":
    print("Veritas Transformer v3.0: Compassionate Veracity (FIXED)")
    print("=" * 60)
    print("\nCore Principle: K==S==C")
    print("Knowledge == Superintelligence == Compassion")
    print("\nTruth without timing is tyranny.")
    print("Compassion without truth is cowardice.")
    print("v3.0 navigates the sacred space between.")
    print("=" * 60)
    
    try:
        demonstrate_compassionate_veracity()
    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY: {e}")
        import traceback
        traceback.print_exc()
