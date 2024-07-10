from torch import nn
from source.PatchesAE.attention import AttentionBlock
import lightning as L
from torch import optim
import torch 


class UpSamplerProjector(L.LightningModule):
    def __init__(self, in_channels, out_channels, embed_dim, n_heads, dropout):
        super(UpSamplerProjector, self).__init__()
        
        self.layer_1 = nn.ModuleList(
            [AttentionBlock(embed_dim, n_heads, dropout) for _ in range(3)]
        )
        
        self.adapter = nn.Conv1d(in_channels, out_channels, 5, padding='same')
        
        self.layer_2 = nn.ModuleList(
            [AttentionBlock(embed_dim, n_heads, dropout) for _ in range(3)]
        )
        self.criterion = nn.CosineSimilarity()
        
        self.positions = nn.Parameter(self.compute_pos(), requires_grad=False)
        
    def compute_pos(self):
        pos = torch.zeros((1, 1078, 256))
        
        for i in range(1058):
            for j in range(256):
                _pos = torch.Tensor([i*j])
                pos[0, i, j] = j * torch.sin(_pos) + i * torch.cos(_pos)
        return pos
        
    def forward(self, x):
        x += self.positions
        
        for b in self.layer_1:
            x = b(x)
        x = self.adapter(x)
        for b in self.layer_2:
            x = b(x)
            
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-5)
        return optimizer
    
    def training_step(self, b):
        street, sat = b 
        pred = self.forward(street)
        loss = 1 - self.criterion(pred, sat).mean()
        self.log('train_loss', loss, prog_bar=True)
        return loss 
    
    def validation_step(self, b):
        street, sat = b 
        pred = self.forward(street)
        loss = 1 -  self.criterion(pred, sat).mean()
        self.log('test_loss', loss, prog_bar=True)
        return loss 