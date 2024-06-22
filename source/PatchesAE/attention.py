from torch import nn 


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(AttentionBlock, self).__init__()
        
        # encoding query, key and value
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # multi head self attention block
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # first layer normalization block
        self.norm_1 = nn.LayerNorm(embed_dim)
        
        # multi layer perceptron block
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU()
        )
        
        # second layer normalization block
        self.norm_2 = nn.LayerNorm(embed_dim)
        
        # initialize weights in a robust way
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
    
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        attention, _ = self.mhsa(q, k, v)
        x = self.norm_1(attention + x)
        
        out = self.mlp(x) + x 
        out = self.norm_2(out)
        return out