import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchmetrics.functional.classification.precision_recall import precision_recall
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.f_beta import f1
import pytorch_lightning as pl
from ssl_byol import model



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
        hidden = self.encoder(T2)
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


class CM_Encoder(nn.Module):
    def __init__(self, cw):
        super().__init__()
        self.cw = cw
        encoder = model.learner
        
        encoder.load_state_dict(torch.load('./model/0.66_0.51_0.72.pth'))
        # encoder.load_state_dict(torch.load('./model/0.7_0.49_0.65.pth'))
        # print(encoder)

        # for parameter in encoder.net._conv_stem.parameters():
        #     parameter.requires_grad = False
        # for parameter in encoder.net._bn0.parameters():
        #     parameter.requires_grad = False
        # for parameter in encoder.net._blocks[:-5].parameters():
        #     parameter.requires_grad = False

        # for parameter in encoder.parameters():
        #     parameter.requires_grad = False
        

        d_model = 1408
        self.d_model = d_model
        self.netwrapper = NetWrapper(encoder, layer='net._conv_head')
        # self.netwrapper = NetWrapper(encoder, layer='net._blocks.22')
        self.conn = torch.nn.Conv3d(1408, d_model, (3, 1, 1))
        self.block = nn.Sequential(
            nn.BatchNorm2d(d_model),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.proj = nn.Sequential(
            # nn.Dropout(.5),
            # nn.Linear(d_model, d_model),
            # nn.SiLU(),
            nn.Dropout(.3),
            nn.Linear(d_model, 2),
        )

    def forward(self, T1_HB, T2, out, label=None,):
        bsz = T2.size(0)

        t1 = self.netwrapper(T1_HB)
        t2 = self.netwrapper(T2)
        out = self.netwrapper(out)

        fmap = torch.stack([t1, t2, out], dim=2)
        # print(fmap.shape)
        fmap = self.conn(fmap).squeeze(2)

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
        # self.encoder.load_state_dict(torch.load('./model/0.66_0.51_0.72.pth'))
        self.encoder.load_state_dict(torch.load('./model/0.55_0.72_0.59.pth'))

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
        # else:
        #     self.encoder = ssl_encoder(self.cw)
        else:
            self.encoder = encoder(self.cw)

        if kargs.get('p'):print(self.encoder)

    def forward(self, T1_HB, T2, out, label=None,):
            return self.encoder(T1_HB, T2, out, label=label)

    def training_step(self, batch, _):
        loss, _, = self.forward(**batch)
        self.log("loss", loss, on_step=False, prog_bar=True)
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
            metrics, on_epoch=True, prog_bar=True, on_step=False
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
