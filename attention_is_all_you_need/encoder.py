import torch
from torch import nn
import math
 
 
class MultiHeadAttention(nn.Module):
 
    def __init__(self, d_model: int, n_heads: int, dropout_proba: float):
        super().__init__()
 
        self.dropout_proba = dropout_proba
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
 
    # From the paper:
    # Scaled Dot-Product Attention(Q, K, V ) = softmax(QKT / √dk)V
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        batch, seq_len = x.size(0), x.size(1)
 
        Q = self.Q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.K(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.V(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
 
        # [batch, d_model, seq_len, seq_len]
        attn_logits = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
 
        # Not explicitly in the paper, but in practice we need to handle padding mask
        # to support variable length sequences
        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
 
        attn_weights = nn.functional.softmax(attn_logits, -1)
 
        # note: while not present in the latest paper version, dropping weights here is still recommended
        # and was present through versions 1-4 of the paper, under "Attention Dropout" section https://arxiv.org/pdf/1706.03762v4/pdf
        attn_weights_dropped = nn.functional.dropout(
            attn_weights, self.dropout_proba, training=self.training
        )
 
        attention_scores = attn_weights_dropped @ V
        attention_scores = (
            attention_scores.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        )
 
        return self.W_O(attention_scores)
 
 
class TransformerLayer(nn.Module):
 
    def __init__(self, d_model: int, n_heads: int, dropout_proba: float):
        super().__init__()
 
        self.dropout_proba = dropout_proba
 
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
 
        # As per paper, we implement two sub-layers.
        # First is a multi head self attention layer
        self.multi_head_self_attn = MultiHeadAttention(d_model, n_heads, dropout_proba)
 
        # Second sub-layer is a position-wise fully connected feed-forward network
        self.FFN = nn.Sequential(
            # defaults will be 512 -> 2048, same as in the paper
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_proba),
            nn.Linear(d_model * 4, d_model),
        )
 
    # x should be of shape [batch, seq_len, d_model]
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # From the paper:
        # MultiHead(Q, K, V ) = Concat(head1, ..., headh)W^O
        multi_head_attn_out = self.add_and_norm(
            self.layer_norm_1,
            nn.functional.dropout(
                self.multi_head_self_attn(x, padding_mask=padding_mask),
                self.dropout_proba,
                training=self.training,
            ),
            x,
        )
 
        # Feed-forward network
        block_out = self.add_and_norm(
            self.layer_norm_2,
            nn.functional.dropout(
                self.FFN(multi_head_attn_out),
                self.dropout_proba,
                training=self.training,
            ),
            multi_head_attn_out,
        )
 
        return block_out
 
    def add_and_norm(self, layer_norm: nn.LayerNorm, input: torch.Tensor, x: torch.Tensor):
        added = torch.add(input, x)
        return layer_norm(added)
 
 
class Encoder(nn.Module):
 
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout_proba: float = 0.1,
    ):
        super().__init__()
 
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout_proba)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(d_model=d_model, n_heads=n_heads, dropout_proba=dropout_proba)
                for _ in range(0, n_layers)
            ]
        )
 
    def forward(self, token_ids: torch.Tensor):
        embeddings = self.token_embedding(token_ids)
 
        # From the paper:
        # In the embedding layers, we multiply those weights by √dmodel.
        embeddings = embeddings * math.sqrt(self.d_model)
 
        pos_encoding = self.get_pos_encoding(embeddings)
        out = self.dropout(embeddings + pos_encoding)
 
        for layer in self.transformer_layers:
            out = layer(out)
 
        return out
 
    # x is [batch, seq_len, d_model]
    # PE(pos,2i)   = sin(pos/10000^(2i/dmodel))
    # PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
    def get_pos_encoding(self, x: torch.Tensor):
        seq_len = x.size(1)
 
        # [seq_len, 1]
        pos = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        # [d_model / 2]
        i = torch.arange(0, self.d_model, 2, dtype=x.dtype, device=x.device)
        div_term = torch.exp(i * -(math.log(10000.0) / self.d_model))
 
        pos_encoding = torch.zeros(seq_len, self.d_model, dtype=x.dtype, device=x.device)
        pos_encoding[:, 0::2] = torch.sin(pos * div_term)
        pos_encoding[:, 1::2] = torch.cos(pos * div_term)
 
        return pos_encoding.unsqueeze(0)
