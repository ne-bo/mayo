import math
from datetime import datetime
from typing import Optional
import numpy as np
import pretrainedmodels
import torch
import torchmetrics.classification
from pytorch_lightning import LightningModule


# import pretrainedmodels


class MayoModel(LightningModule):
    def __init__(
            self,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.resnet = pretrainedmodels.resnet18()
        self.fc = torch.nn.Sequential(torch.nn.Linear(512, 4),
                                      torch.nn.Softmax()
                                      )

        #print('self.resnet ', self.resnet)

    def forward(self, **inputs):
        image = inputs['image'].float()
        features = self.resnet.features(image)
        pooled_features = self.resnet.avgpool(features).reshape(features.shape[0], -1)
        out_cls = self.fc(pooled_features)
        return {'out_cls': out_cls
                }

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        ce_loss = torch.nn.CrossEntropyLoss()
        labels = batch["label"]
        batch_size = len(labels)

        loss = ce_loss(outputs['out_cls'], labels)
        self.logger.experiment.add_scalar("training loss", loss.data.cpu().numpy(), self.current_epoch)
        return {"loss": loss,
                "preds_cls": outputs['out_cls'],
                "labels": labels}

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        ce_loss = torch.nn.CrossEntropyLoss()
        labels = batch["label"]
        val_loss = ce_loss(outputs['out_cls'], labels)
        self.logger.experiment.add_scalar("validation loss", val_loss.data.cpu().numpy(), self.current_epoch)
        self.log('val_loss', val_loss)
        # self.log('val_cont', val_cont)
        return {"loss": val_loss,
                "preds_cls": outputs['out_cls'],
                "labels": labels}

    def training_epoch_end(self, outputs):
        preds_cls = torch.cat([x["preds_cls"] for x in outputs]).detach().cpu().float()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        metric_cls = torchmetrics.classification.F1Score(y_pred=preds_cls.numpy(), y_true=labels.numpy())
        print('metric train '
              ' f1 ', metric_cls,
              )

    def validation_epoch_end(self, outputs):
        preds_cls = torch.cat([x["preds_cls"] for x in outputs]).detach().cpu().float()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        metric_cls = torchmetrics.classification.F1Score(y_pred=preds_cls.numpy(), y_true=labels.numpy())
        print('metric train '
              ' f1 ', metric_cls,
              )
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(preds_cls, labels)
        return {'loss': loss}  # , 'natasha_metric': metric}

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for p in self.resnet.parameters()],  # if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for p in self.fc.parameters()],  # if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                      eps=self.hparams.adam_epsilon)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4], gamma=0.3)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": "val_loss"}
        return [optimizer], [scheduler]