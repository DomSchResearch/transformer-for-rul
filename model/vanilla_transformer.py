"""
Vanilla Transformer Module
"""

import lightning.pytorch as pl
import torch
import numpy as np
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class TransformerLRS(_LRScheduler):
    """
    Custom Transformer Learning Rate Scheduler
    """
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self._calc_lr(self._step_count)
        return [lr] * self.num_param_groups

    def _calc_lr(self, step):
        return self.dim_embed**(-0.5) * min(step**(-0.5), step * self.warmup_steps**(-1.5))

class PositionalEncoding(torch.nn.Module):
    """
    Positional Encoding
    """
    def __init__(self, length, depth):
        super().__init__()

        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pe = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        self.register_buffer('pe', torch.Tensor(pe))

    def forward(self, x):
        """
        Forward pass
        """
        x = x + self.pe
        return x

class ParamExtraction(torch.nn.Module):
    """
    Parameter Extraction
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        """
        Forward pass
        """
        t_min = torch.unsqueeze(torch.min(x, dim=1).values, dim=1)
        t_max = torch.unsqueeze(torch.max(x, dim=1).values, dim=1)
        t_mean = torch.unsqueeze(torch.mean(x, dim=1), dim=1)
        t_std = torch.unsqueeze(torch.std(x, dim=1), dim=1)
        ret = torch.cat([x[:, -2:, :], t_min, t_max, t_mean, t_std], dim=1)
        return ret

class Encoder(torch.nn.Module):
    """
    Encoder
    """
    def __init__(self, dimv, dimatt, n_heads, drop):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(self, x, mask=None):
        """
        Forward pass
        """
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=mask)
        x = self.ln2(a + x)
        a = self.ffn2(torch.nn.ReLU()(self.ffn1(x)))
        return x + a

class Decoder(torch.nn.Module):
    """
    Decoder
    """
    def __init__(self, dimv, dimatt, n_heads, drop):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn1 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln2 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.attn2 = torch.nn.MultiheadAttention(
            dimatt,
            n_heads,
            drop,
            batch_first=True
        )
        self.ln3 = torch.nn.LayerNorm(dimv, eps=1e-5)
        self.ffn1 = torch.nn.Linear(dimv, dimv)
        self.ffn2 = torch.nn.Linear(dimv, dimv)

    def forward(self, x, enc):
        """
        Forward pass
        """
        a = self.ln1(x)
        a, _ = self.attn1(a, a, a, key_padding_mask=None)
        x = self.ln2(a + x)
        a, _ = self.attn2(x, enc, enc, key_padding_mask=None)
        x = self.ln3(a + x)
        a = self.ffn2(torch.nn.ReLU()(self.ffn1(x)))
        return x + a

class VanTransLitModule(pl.LightningModule):
    """
    Vanilla Transformer Module
    """
    def __init__(self):
        super().__init__()

        self.input_size = (40, 17)
        self.d_model = 64
        self.heads = 4
        self.nencoder = 2
        self.ndecoder = 1
        self.dim_val = self.d_model
        self.dec_l = 6
        self.output_size = 1

        self.encoder = torch.nn.ModuleList(
            [Encoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.nencoder)])

        self.decoder = torch.nn.ModuleList(
            [Decoder(self.dim_val, self.d_model, self.heads, 0)
             for _ in range(self.ndecoder)])

        self.pos = torch.nn.ModuleList(
            [PositionalEncoding(self.input_size[0], self.dim_val)])

        self.decinp = torch.nn.ModuleList(
            [ParamExtraction()]
        )

        self.encemb = torch.nn.Linear(self.input_size[1], self.dim_val)
        self.decemb = torch.nn.Linear(self.input_size[1], self.dim_val)

        self.ln1 = torch.nn.LayerNorm(self.dim_val, eps=1e-5)
        self.out = torch.nn.Linear(self.dec_l*self.dim_val,
                                   self.output_size)

    def forward(self, x):
        """
        Forward pass
        """
        e = self.encoder[0](self.pos[0](self.encemb(x)))

        for enc in self.encoder[1:]:
            e = enc(e)

        p = self.ln1(e)

        d = self.decoder[0](self.decemb(self.decinp[0](x)), p)
        for dec in self.decoder[1:]:
            d = dec(d, p)

        x = self.out(torch.nn.ReLU()(d.flatten(start_dim=1)))

        return x.flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        _, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_RMSE', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        preds, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_RMSE', loss, sync_dist=True)

        return preds

    def test_step(self, batch, batch_idx):
        """
        Test step
        """
        _, loss = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_RMSE', loss, sync_dist=True)

    def predict_step(self, *args, **kwargs):
        """
        Predict step
        """
        x, y = args[0]
        return self(x)

    def configure_optimizers(self):
        """
        Configure optimizer
        """
        optimizer = Adam(
            params=self.parameters(),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        lr_scheduler = TransformerLRS(optimizer, self.d_model, 4000)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": 'val_RMSE'}

    def _get_preds_loss_accuracy(self, batch):
        """
        Get predictions, loss and accuracy
        """
        x, y = batch
        preds = self(x)
        loss = torch.sqrt(torch.nn.MSELoss()(preds, torch.unsqueeze(y, 1)))
        return preds, loss
