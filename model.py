import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, input_seq, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, input_seq, d_model))

    def forward(self, x):
        x = x + self.position_embedding
        return x


class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MHA, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self.scores = None

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q, k, v = (self.split(x, (self.nhead, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.dropout(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge(h, 2)
        self.scores = scores
        return h, scores

    def split(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)

    def merge(self, x, n_dims):
        s = x.size()
        assert 1 < n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.ff2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        x = self.ff2(F.gelu(self.ff1(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = MHA(d_model=d_model, nhead=nhead, dropout=dropout)
        self.linproj = nn.Linear(in_features=d_model, out_features=d_model)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.ff = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, scores = self.attn(self.norm1(x))
        h = self.dropout(self.linproj(h))
        x = x + h
        h = self.dropout(self.ff(self.norm2(x)))
        x = x + h
        return x, scores


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        scores = []
        for block in self.blocks:
            x, score = block(x)
            scores.append(score)
        return x, scores


class ViT(nn.Module):

    def __init__(self,
                 patches,  # Patch size: height width
                 d_model,  # Token Dim
                 d_ff,  # Feed Forward Dim
                 num_heads,  # Num MHA
                 num_layers,  # Num Transformer Layers
                 dropout,  # Dropout rate
                 image_size,  # channels, height, width
                 num_classes,  # Dataset Categories
                 ):
        super(ViT, self).__init__()

        self.image_size = image_size

        # ---- 1 Patch Embedding ---
        c, h, w = image_size  # image sizes

        ph, pw= patches  # patch sizes

        n, m = h // ph, w // pw
        seq_len = n * m  # number of patches

        # Patch embedding
        """
        The original Vision Transformer uses Linear patching, but We implement Conv2D patching instead of Linear
         patching with the same number of parameters. It introduces inductive bias to helps model train in small
         scale datasets.
        """
        self.patch_embedding = nn.Conv2d(in_channels=c, out_channels=d_model, kernel_size=(ph, pw), stride=(ph, pw))

        # Class token
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.position_embedding = PositionEmbedding(input_seq=(seq_len + 1), d_model=d_model)

        # Transformer
        """
        we follow the original Vision Transformer in employing transformer encoder 
        """
        self.transformer = TransformerEncoder(num_layers=num_layers, d_model=d_model, nhead=num_heads,
                                              d_ff=d_ff, dropout=dropout)

        # Classifier head
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp = nn.Linear(in_features=d_model, out_features=num_classes)

    def forward(self, x):
        b, c, ph, pw = x.shape

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        x = torch.cat((self.class_embedding.expand(b, -1, -1), x), dim=1)

        x = self.position_embedding(x)

        x, scores = self.transformer(x)
        x = self.norm(x)[:, 0]
        x = self.mlp(x)
        return x, scores

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ViT(
        patches=(16, 16),
        d_model=768,
        d_ff=3072,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        image_size=(3, 384, 384),
        num_classes=1000,
    )
    print(f'Parameter Number: {model.count_params()}')

    input_example = torch.zeros((1,3,384,384))

    outputs, scores = model(input_example)
