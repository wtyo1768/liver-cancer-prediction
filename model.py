import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchmetrics.functional.classification.precision_recall import precision_recall
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.f_beta import f1
import pytorch_lightning as pl
from ssl_byol import model
import torch.nn.functional as F



class encoder(nn.Module):
    def __init__(self, cw):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained('efficientnet-b2',)

        for parameter in self.encoder._conv_stem.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder._bn0.parameters():
            parameter.requires_grad = False
        for parameter in self.encoder._blocks[:-5].parameters():
            parameter.requires_grad = False

        self.cw = cw
        self.d_proj = self.encoder._fc.out_features
        self.proj = nn.Sequential(
            nn.Linear(self.d_proj, 2),
        )


    def forward(self, T1_HB, T2, out, label=None,):
        hidden = self.encoder(out)
        logits = self.proj(hidden)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.cw).to(logits)
            )
            loss = loss_fct(logits, label.view(-1))
            return loss, logits
        return logits


class NetWrapper(nn.Module):
    def __init__(self, net, layer ='_conv_head'):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hook_registered = False
        self.hidden = {}

    def _find_layer(self):
        modules = dict([*self.net.named_modules()])
        return modules.get(self.layer, None)

    def _hook(self, _, input, output):
        device = input[0].device
        # print(device)
        self.hidden[device] = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)
        return representation


class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, groups=group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            # dw
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)


    def forward(self, old_x):
        x = old_x
        x_proj = old_x

        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_1(x)
        
        return F.silu(x + x_proj)


    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class CM_Encoder(nn.Module):
    def __init__(self, cw):
        super().__init__()
        self.cw = cw
        encoder = model.learner
        
        encoder.load_state_dict(torch.load('./model/0.66_0.51_0.72.pth'))        

        d_model = 1408
        self.d_model = d_model
        self.netwrapper = NetWrapper(encoder, layer='net._conv_head')

        group = 8
        self.conv3d = torch.nn.Conv3d(1408, d_model, (3, 1, 1), groups=group)

        self.shuffle_block = ShuffleV1Block(1408, 1408, group=group,    
                                            mid_channels=1408, ksize=1, stride=1)
        self.shuffle_block2 = ShuffleV1Block(1408, 1408, group=group,
                                            mid_channels=1408, ksize=1, stride=1)
        self.shuffle_block3 = ShuffleV1Block(1408, 1408, group=group,
                                            mid_channels=1408, ksize=1, stride=1)

        self.block = nn.Sequential(
            # nn.BatchNorm2d(d_model),
            nn.AdaptiveAvgPool2d(output_size=1),
        )   
        self.proj = nn.Sequential(
            nn.Dropout(.3),
            nn.Linear(d_model, 2),
        )

    def forward(self, T1_HB, T2, out, label=None,):
        bsz = T2.size(0)

        t1 = self.netwrapper(T1_HB)
        t2 = self.netwrapper(T2)
        out = self.netwrapper(out)

        fmap = torch.stack([t1, t2, out], dim=2)
        
        fmap = self.conv3d(fmap).squeeze(2)
        fmap = self.shuffle_block(fmap)
        fmap = self.shuffle_block2(fmap)
        fmap = self.shuffle_block3(fmap)

        hidden = self.block(fmap)
        hidden= hidden.view(bsz, self.d_model)

        logits = self.proj(hidden)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.cw).to(logits)
            )
            loss = loss_fct(logits, label.view(-1))
            return loss, logits
        return logits


class ssl_encoder(nn.Module):
    def __init__(self, cw):
        super().__init__()
        self.cw = cw
        self.encoder = model.learner  
        self.encoder.load_state_dict(torch.load('./model/t1_0.62_0.51_0.57.pth'))

        self.pj =  nn.Sequential(
            nn.Dropout(.3),
            nn.Linear(256, 2),
        )


    def forward(self, T1_HB, T2, out, label=None,):
        emb, _ = self.encoder(T1_HB ,return_embedding=True)
        logits = self.pj(emb)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.cw).to(logits)
            )
            loss = loss_fct(logits, label.view(-1))
            return loss, logits
        return logits


class cls(pl.LightningModule):
    def __init__(self, hparams=None, ssl_pretrain=False, enc=None, **kargs):
        super().__init__()
        self.cw = kargs.get('class_weight')
        if hparams:
            self.lr = hparams.LR
        if enc=='ca':
            self.encoder = CM_Encoder(self.cw)
        else:
            self.encoder = ssl_encoder(self.cw)

    def forward(self, T1_HB, T2, out, label=None,):
            return self.encoder(T1_HB, T2, out, label=label)

    def training_step(self, batch, _):
        loss, _, = self.forward(**batch)
        self.log("loss", loss, on_step=False, prog_bar=False)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, logits = self.forward(**batch)

        label = batch['label'].view(-1)
        pred = torch.max(logits, dim=1).indices

        pr = precision_recall(pred, label, num_classes=1, is_multiclass=False)
        metrics = {
            'val_loss': loss, 
            'val_acc':  accuracy(pred, label),
            'f1':       f1(pred, label, num_classes=1), 
            'recall':   pr[1],
            'prec':     pr[0],
        }
        self.log_dict(
            metrics, on_epoch=True, prog_bar=False, on_step=False
        )
        return metrics


    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_schedulers  = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode='min',
                factor=0.2, patience=2,
                min_lr=1e-6, verbose=True
            ), 
            'monitor': 'val_loss'
        }
        return [self.opt], [lr_schedulers]


    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
        ):
        # print(len(self.train_dataloader))
        warmup_step = 50.0
        if self.trainer.global_step < warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / warmup_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        optimizer.step(closure=opt_closure)
        optimizer.zero_grad()
