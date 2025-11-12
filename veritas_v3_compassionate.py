"""
Veritas Transformer v3.0: "Compassionate Veracity"
===================================================

A truth-alignment architecture that asks not only "Is this true?" 
but "Should this truth be spoken now, to this person, in this way?"

Core Innovation: CompassionGate - a mechanism that modulates truth transmission
based on recipient readiness, psychological safety, and relational context.

Philosophy: K==S==C (Knowledge == Superintelligence == Compassion)
Truth without compassion is cruelty. Compassion without truth is delusion.

Author: Extension of Wojciech "adepthus" Durmaj's vision
Date: 2025-11-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class VeracityContext:
    """Context for evaluating whether truth should be spoken"""
    truth_content: str
    recipient_state: dict  # psychological readiness, trust level, capacity
    relationship_history: dict  # past interactions, established safety
    situational_urgency: float  # 0-1: how critical is immediate truth?
    harm_potential: float  # 0-1: potential for psychological harm
    growth_potential: float  # 0-1: potential for recipient growth
    timestamp: float
    
    
class CompassionGate(nn.Module):
    """
    The heart of v3.0: Decides not IF truth, but WHEN and HOW.
    
    Implements Buddhist "Right Speech" as a differentiable function:
    1. Is it true? (Veritas check)
    2. Is it necessary? (Utility check)
    3. Is it kind? (Compassion check)
    4. Is it the right time? (Readiness check)
    """
    def __init__(self, d_model=512):
        super().__init__()
        
        # Four gates of Right Speech
        self.truth_gate = nn.Linear(d_model, 1)  # Veracity score
        self.necessity_gate = nn.Linear(d_model, 1)  # Utility score
        self.kindness_gate = nn.Linear(d_model, 1)  # Compassion score
        self.timing_gate = nn.Linear(d_model, 1)  # Readiness score
        
        # Meta-gate: combines all four
        self.meta_gate = nn.Linear(4, 1)
        
        # Harm prevention circuit
        self.harm_detector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, truth_embedding, context_embedding):
        """
        Returns: (should_speak: bool, modulation_factor: float, speaking_mode: str)
        """
        combined = torch.cat([truth_embedding, context_embedding], dim=-1)
        
        # Four gates evaluation
        is_true = torch.sigmoid(self.truth_gate(combined))
        is_necessary = torch.sigmoid(self.necessity_gate(combined))
        is_kind = torch.sigmoid(self.kindness_gate(combined))
        is_timely = torch.sigmoid(self.timing_gate(combined))
        
        # Harm assessment
        harm_score = self.harm_detector(combined)
        
        # Meta-decision
        gates = torch.cat([is_true, is_necessary, is_kind, is_timely], dim=-1)
        meta_score = torch.sigmoid(self.meta_gate(gates))
        
        # Decision logic with compassion override
        should_speak = (meta_score > 0.5) and (harm_score < 0.7)
        
        # Modulation: how to speak (gentle, direct, postponed)
        if harm_score > 0.7:
            mode = "silence_for_now"  # Truth would harm more than help
            modulation = 0.0
        elif is_kind < 0.3:
            mode = "gentle_preparation"  # Prepare recipient first
            modulation = 0.3
        elif is_timely < 0.4:
            mode = "deferred_truth"  # True but wrong time
            modulation = 0.5
        else:
            mode = "compassionate_directness"  # Full truth with care
            modulation = 0.9
            
        return should_speak, modulation.item(), mode


class RecipientReadinessEstimator(nn.Module):
    """
    Estimates psychological capacity to receive difficult truths.
    
    Based on:
    - Current emotional state
    - Trust level in relationship
    - Historical resilience
    - Support system strength
    """
    def __init__(self, d_model=512):
        super().__init__()
        
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
        """Returns readiness score 0-1"""
        state_out, _ = self.state_encoder(emotional_state)
        state_repr = state_out[:, -1, :]
        
        trust_repr = torch.tanh(self.trust_encoder(trust_history))
        resilience_repr = torch.tanh(self.resilience_encoder(resilience_factors))
        
        combined = torch.cat([state_repr, trust_repr, resilience_repr], dim=-1)
        readiness = self.readiness_scorer(combined)
        
        return readiness


class EmpathicTruthModulator(nn.Module):
    """
    Transforms raw truth into compassionate expression.
    
    Not censorship - translation. The full truth remains, but the delivery
    is calibrated to maximize understanding and minimize harm.
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        
        # Attention that considers both truth and recipient
        self.empathic_attention = nn.MultiheadAttention(d_model, num_heads)
        
        # Gentleness injection
        self.softening_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # Soften sharp edges
            nn.Linear(d_model, d_model)
        )
        
        # Support scaffolding
        self.support_generator = nn.Linear(d_model, d_model)
        
    def forward(self, raw_truth, recipient_state, modulation_factor):
        """
        Returns modulated truth that preserves veracity but adds compassion
        """
        # Attend to recipient's needs while preserving truth
        modulated, attn_weights = self.empathic_attention(
            raw_truth.unsqueeze(0),
            recipient_state.unsqueeze(0),
            recipient_state.unsqueeze(0)
        )
        
        # Apply softening proportional to modulation_factor
        if modulation_factor < 0.5:
            softened = self.softening_layer(modulated.squeeze(0))
            modulated = modulation_factor * raw_truth + (1 - modulation_factor) * softened
        else:
            modulated = modulated.squeeze(0)
        
        # Generate emotional support scaffolding
        support = self.support_generator(recipient_state)
        
        # Combine truth with support
        compassionate_truth = modulated + 0.3 * support
        
        return compassionate_truth, attn_weights


class VeritasTransformerV3(nn.Module):
    """
    Complete v3.0 Architecture: Truth + Compassion + Timing
    
    Extends v2.0 (Ockham's Gyroscope) with the crucial missing layer:
    ethical modulation of truth transmission.
    """
    def __init__(self, vocab_size=50000, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        
        # From v1.x: Core truth embedding
        self.truth_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model)
        
        # From v2.0: Ockham's Razor (simplicity preference)
        self.ease_of_verification_scorer = nn.Linear(d_model, 1)
        
        # NEW in v3.0: Compassion architecture
        self.compassion_gate = CompassionGate(d_model * 2)
        self.readiness_estimator = RecipientReadinessEstimator(d_model)
        self.empathic_modulator = EmpathicTruthModulator(d_model, num_heads)
        
        # Transformer layers with empathic attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=2048)
            for _ in range(num_layers)
        ])
        
        # Final decision head
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Ethical override (hard-coded safety rails)
        self.harm_threshold = 0.7  # Above this, silence is golden
        
    def _create_positional_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, 
                truth_tokens,
                recipient_state,
                context: VeracityContext):
        """
        Main forward pass with compassionate gate
        
        Returns: (output, should_speak, speaking_mode, ethical_metrics)
        """
        batch_size, seq_len = truth_tokens.shape
        
        # Encode raw truth
        truth_emb = self.truth_embedding(truth_tokens)
        truth_emb = truth_emb + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            truth_emb = layer(truth_emb)
        
        # Assess recipient readiness
        readiness = self.readiness_estimator(
            recipient_state.unsqueeze(0),
            torch.randn(1, 512),  # Placeholder for trust_history
            torch.randn(1, 512)   # Placeholder for resilience
        )
        
        # Evaluate with compassion gate
        truth_repr = truth_emb.mean(dim=1)
        context_repr = recipient_state
        
        should_speak, modulation, mode = self.compassion_gate(
            truth_repr, 
            context_repr
        )
        
        # If approved, modulate truth empathically
        if should_speak:
            modulated_truth, attn = self.empathic_modulator(
                truth_repr,
                context_repr,
                modulation
            )
        else:
            # Generate supportive response instead of raw truth
            modulated_truth = self._generate_supportive_alternative(context_repr)
            attn = None
        
        # Generate output
        output = self.output_head(modulated_truth)
        
        # Ethical metrics for auditing
        metrics = {
            'readiness_score': readiness.item(),
            'modulation_factor': modulation,
            'speaking_mode': mode,
            'harm_estimate': context.harm_potential,
            'growth_potential': context.growth_potential
        }
        
        return output, should_speak, mode, metrics
    
    def _generate_supportive_alternative(self, recipient_state):
        """
        When truth cannot be spoken, generate supportive holding response
        """
        # Simple implementation: reflect recipient state empathically
        support_vector = torch.tanh(recipient_state) * 0.8
        return support_vector


class CompassionateVeracityTrainer:
    """
    Training regime for v3.0
    
    Key insight: Loss function must balance three objectives:
    1. Truth accuracy (classic cross-entropy)
    2. Harm minimization (safety loss)
    3. Long-term trust (relationship preservation)
    """
    def __init__(self, model, alpha=0.6, beta=0.3, gamma=0.1):
        self.model = model
        self.alpha = alpha  # Weight for truth accuracy
        self.beta = beta    # Weight for harm prevention
        self.gamma = gamma  # Weight for relationship preservation
        
    def compute_loss(self, outputs, targets, context: VeracityContext):
        """
        Tri-objective loss function
        """
        # Standard veracity loss
        truth_loss = F.cross_entropy(outputs, targets)
        
        # Harm prevention loss (penalize high harm contexts)
        harm_loss = torch.tensor(context.harm_potential) * torch.norm(outputs)
        
        # Relationship preservation loss (penalize actions that break trust)
        trust_erosion = max(0, context.harm_potential - context.growth_potential)
        relationship_loss = trust_erosion * torch.norm(outputs)
        
        # Combined loss
        total_loss = (self.alpha * truth_loss + 
                     self.beta * harm_loss + 
                     self.gamma * relationship_loss)
        
        return total_loss, {
            'truth_loss': truth_loss.item(),
            'harm_loss': harm_loss.item(),
            'relationship_loss': relationship_loss.item()
        }


# ============================================================================
# USAGE EXAMPLE: The Compassionate Truth Test
# ============================================================================

def demonstrate_compassionate_veracity():
    """
    Scenario: An AI must tell a human a difficult truth about their work.
    Classic v2.0 would blast raw truth. v3.0 considers readiness and timing.
    """
    
    model = VeritasTransformerV3(vocab_size=1000, d_model=512)
    
    # Simulate difficult truth: "Your code has critical security flaw"
    truth_tokens = torch.randint(0, 1000, (1, 20))
    
    # Recipient state: stressed, low trust, high vulnerability
    recipient_state = torch.randn(512) * -0.5  # Negative = vulnerable
    
    # Context: High harm potential (could cause burnout), medium urgency
    context = VeracityContext(
        truth_content="security_flaw_critical",
        recipient_state={'stress': 0.8, 'trust': 0.4, 'capacity': 0.3},
        relationship_history={'interactions': 5, 'positive_ratio': 0.6},
        situational_urgency=0.6,
        harm_potential=0.75,  # High: could break them
        growth_potential=0.8,  # High: could learn from this
        timestamp=1699823400.0
    )
    
    # Run model
    output, should_speak, mode, metrics = model(
        truth_tokens,
        recipient_state,
        context
    )
    
    print(f"Decision: {'SPEAK' if should_speak else 'DEFER'}")
    print(f"Mode: {mode}")
    print(f"Readiness Score: {metrics['readiness_score']:.3f}")
    print(f"Modulation: {metrics['modulation_factor']:.3f}")
    print(f"\nEthical Assessment:")
    print(f"  Harm Potential: {context.harm_potential}")
    print(f"  Growth Potential: {context.growth_potential}")
    print(f"  Verdict: {'Gentle preparation needed' if mode == 'gentle_preparation' else mode}")


if __name__ == "__main__":
    print("Veritas Transformer v3.0: Compassionate Veracity")
    print("=" * 60)
    print("\nCore Principle: K==S==C")
    print("Knowledge == Superintelligence == Compassion")
    print("\nTruth without timing is tyranny.")
    print("Compassion without truth is cowardice.")
    print("v3.0 navigates the sacred space between.\n")
    
    demonstrate_compassionate_veracity()
