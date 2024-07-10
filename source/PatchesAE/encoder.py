import torch 
from torch import nn
from source.PatchesAE.attention import AttentionBlock


class Encoder(nn.Module):
    def __init__(self, embed_dim, in_channels, patch_size, depth, num_heads, dropout):
        super(Encoder, self).__init__()
        
        self.embedder = nn.Linear(patch_size * patch_size * in_channels, embed_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.out = nn.Tanh()
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=2)
        embeddings = self.embedder(x)   # BSZ x NUM_PATCHES x EMBED_DIM
        
        for b in self.transformer_blocks:
            embeddings = b(embeddings)
        encodings = self.out(embeddings)
        return encodings