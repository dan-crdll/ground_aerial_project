import torch 
from torch import nn 
from source.PatchesAE.attention import AttentionBlock


class Decoder(nn.Module):
    def __init__(self, embed_dim, in_channels, patch_size, depth, num_heads, dropout):
        super(Decoder, self).__init__() 
        
        self.transformer_blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, dropout) for _ in range(depth)]
        )
        
        self.out_layer = nn.Sequential(
            nn.Linear(embed_dim, in_channels * patch_size ** 2, bias=False),
            nn.Unflatten(2, (in_channels, patch_size, patch_size)),
            nn.Tanh()
        )
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
        
    def forward(self, x):
        for b in self.transformer_blocks:
            x = b(x)
        x = self.out_layer(x)
        return x