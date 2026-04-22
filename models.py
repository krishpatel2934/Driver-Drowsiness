# ================================================================
# models.py — BiGRU + MultiHead Attention (Final Model)
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAdditiveAttention(nn.Module):
    """
    Multi-head attention with additive (tanh) scoring.
    Each head: score = W2 * tanh(W1 * h_t)  → softmax → weighted sum
    Multiple heads learn different temporal patterns simultaneously.
    Outputs concatenated across heads, projected back to hidden_size.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.W1       = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.W2       = nn.Linear(self.head_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, rnn_out):
        B, T, H = rnn_out.shape
        proj    = torch.tanh(self.W1(rnn_out))
        proj    = proj.view(B, T, self.num_heads, self.head_dim)
        scores  = self.W2(proj).squeeze(-1).permute(0, 2, 1)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        rnn_h   = rnn_out.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        ctx = torch.bmm(
            weights.unsqueeze(2).reshape(B * self.num_heads, 1, T),
            rnn_h.contiguous().reshape(B * self.num_heads, T, self.head_dim)
        ).view(B, self.num_heads, self.head_dim).view(B, H)
        ctx   = self.out_proj(ctx)
        avg_w = weights.mean(dim=1)
        return ctx, avg_w


def build_fc_head(in_size, num_classes=3, dropout=0.3):
    return nn.Sequential(
        nn.Linear(in_size, 128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, 64),      nn.ReLU(), nn.Dropout(dropout * 0.5),
        nn.Linear(64, num_classes)
    )


class BiLSTM_MHA(nn.Module):
    """BiLSTM + Multi-Head Additive Attention. Same backbone as v2, upgraded attention."""
    def __init__(self, input_size=16, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3, num_heads=4):
        super().__init__()
        out = hidden_size * 2
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
        self.attention = MultiHeadAdditiveAttention(out, num_heads, dropout=0.1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = build_fc_head(out, num_classes, dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, _ = self.attention(out)
        return self.fc(self.dropout(ctx))

    def predict_score(self, x):
        self.eval()
        with torch.no_grad():
            p = torch.softmax(self.forward(x), dim=-1)
        return (p[:,1]*50 + p[:,2]*100).cpu().numpy(), p.cpu().numpy()


class BiGRU_MHA(nn.Module):
    """BiGRU + Multi-Head Additive Attention. Same backbone as v3, upgraded attention."""
    def __init__(self, input_size=16, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3, num_heads=4):
        super().__init__()
        out = hidden_size * 2
        self.rnn       = nn.GRU(input_size, hidden_size, num_layers,
                                 batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = MultiHeadAdditiveAttention(out, num_heads, dropout=0.1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = build_fc_head(out, num_classes, dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        ctx, _ = self.attention(out)
        return self.fc(self.dropout(ctx))

    def predict_score(self, x):
        self.eval()
        with torch.no_grad():
            p = torch.softmax(self.forward(x), dim=-1)
        return (p[:,1]*50 + p[:,2]*100).cpu().numpy(), p.cpu().numpy()


MODEL_REGISTRY = {'bilstm_mha': BiLSTM_MHA, 'bigru_mha': BiGRU_MHA}

def get_model(name, input_size=16, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3, num_heads=4):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown: {name}. Choose: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](input_size, hidden_size, num_layers,
                                 num_classes, dropout, num_heads)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_path, device='cpu'):
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['model_config']
    model  = get_model(ckpt['model_name'], **config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    loso = ckpt.get('loso_f1','N/A')
    print(f"Loaded {ckpt['model_name'].upper()} | F1: {loso:.3f if isinstance(loso,float) else loso}")
    return model, ckpt

if __name__ == '__main__':
    x = torch.randn(2, 600, 16)
    for name, cls in MODEL_REGISTRY.items():
        m   = cls(input_size=16, num_heads=4)
        out = m(x)
        print(f"{name}: params={count_params(m):,}  output={out.shape}")
