import gc
from typing import Optional

import pretrainedmodels
import torch
import torchmetrics.classification
from pytorch_lightning import LightningModule


# import pretrainedmodels


class MayoModel(LightningModule):
    def __init__(
            self,
            learning_rate: float = 3e-4,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 1,
            eval_batch_size: int = 1,
            eval_splits: Optional[list] = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

        # self.backbone_64 = pretrainedmodels.resnet18()
        # self.backbone_128 = pretrainedmodels.resnet18()
        # self.backbone_256 = pretrainedmodels.resnet18()
        # self.backbone_384 = pretrainedmodels.resnet18()
        self.backbone_512 = pretrainedmodels.resnet50()
        # print('self.backbone ', self.backbone)
        self.reduce_dim_512 = torch.nn.Sequential(torch.nn.Linear(2048, 1024),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(1024, 512),
                                                  torch.nn.ReLU(), )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.ReLU(),
            torch.nn.Softmax()
        )

        # print('self.resnet ', self.resnet)

    def get_pooled_features(self, boxes, backbone, batch_size, number_of_boxes):
        features_ = backbone.features(boxes)
        pooled_ = backbone.avgpool(features_).reshape(features_.shape[0], -1)
        # print('pooled_ ', pooled_.shape)
        pooled_ = pooled_.view(batch_size, number_of_boxes, -1)
        # print('pooled_ ', pooled_.shape)
        pooled_ = torch.mean(pooled_, dim=1, keepdim=False)
        # print('pooled_ ', pooled_.shape)
        return pooled_

    def forward(self, **inputs):
        # image = inputs['image'].float().to(self.device)
        box_pooled_features = []
        # print('inputs[64] ', inputs['64'].shape) # batch_size, 20, 3, 64, 64
        # print('inputs[512]', inputs['512'])
        batch_size = inputs['512'].shape[0]

        # boxes_64 = inputs['64'].reshape(-1, 3, 64, 64).float().to(self.device)
        # boxes_128 = inputs['128'].reshape(-1, 3, 128, 128).float().to(self.device)
        # boxes_256 = inputs['256'].reshape(-1, 3, 256, 256).float().to(self.device)
        # boxes_384 = inputs['384'].reshape(-1, 3, 384, 384).float().to(self.device)/255.0
        boxes_512 = inputs['512'].reshape(-1, 3, 512, 512).float().to(self.device) / 255.0

        # pooled_64 = self.get_pooled_features(boxes_64, self.backbone_64, batch_size, number_of_boxes)
        # pooled_128 = self.get_pooled_features(boxes_128, self.backbone_128, batch_size, number_of_boxes)
        # pooled_256 = self.get_pooled_features(boxes_256, self.backbone_256, batch_size, number_of_boxes)
        # pooled_384 = self.get_pooled_features(boxes_384, self.backbone_384, batch_size, number_of_boxes=inputs['384'].shape[1])
        pooled_512 = self.get_pooled_features(boxes_512, self.backbone_512, batch_size,
                                              number_of_boxes=inputs['512'].shape[1])

        pooled_512 = self.reduce_dim_512(pooled_512)

        average_pooled_features = (  # pooled_64 +
                                      # pooled_128 +
                                      # pooled_256 +
                                      # pooled_384 +
                                      pooled_512) / 1.0

        # resnets
        # features = self.backbone.features(image)
        # pooled_features = self.backbone.avgpool(features).reshape(features.shape[0], -1)

        # efficientnets
        # features = self.backbone.stem(image)
        # features = self.backbone.layers(features)
        # features = self.backbone.features(features)
        # pooled_features = torch.nn.Flatten()(self.backbone.classifier.pooling(features))

        out_cls = self.fc(average_pooled_features)
        gc.collect()
        return {'out_cls': out_cls
                }

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        ce_loss = torch.nn.CrossEntropyLoss()
        labels = batch["label"]
        batch_size = len(labels)

        loss = ce_loss(outputs['out_cls'], labels)
        self.logger.experiment.add_scalar("training loss", loss.data.cpu().numpy(), self.current_epoch)

        gc.collect()
        return {"loss": loss,
                "preds_cls": outputs['out_cls'].detach(),
                "labels": labels.detach()}

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
        metric_cls = torchmetrics.classification.F1Score()
        f1 = metric_cls(torch.argmax(preds_cls, dim=-1), labels)
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(preds_cls, labels)
        print('metric train '
              ' f1 ', f1,
              'train_loss', loss
              )

    def validation_epoch_end(self, outputs):
        preds_cls = torch.cat([x["preds_cls"] for x in outputs]).detach().cpu().float()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        metric_cls = torchmetrics.classification.F1Score()
        # print('preds_cls ', preds_cls, preds_cls.shape)
        # print('torch.argmax(preds_cls, dim=-1) ', torch.argmax(preds_cls, dim=-1), torch.argmax(preds_cls, dim=-1).shape)
        f1 = metric_cls(torch.argmax(preds_cls, dim=-1), labels)
        ce_loss = torch.nn.CrossEntropyLoss()
        val_loss = ce_loss(preds_cls, labels)
        print('metric val '
              ' f1 ', f1,
              ' val_loss ', val_loss
              )
        return {'val_loss': val_loss}  # , 'natasha_metric': metric}

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size
        print('self.total_steps', self.total_steps,
              'tb_size ', tb_size, ' ab_size ', ab_size, ' len(train_loader.dataset) ', len(train_loader.dataset))
        #input()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # {
            #     "params": [p for p in self.backbone_64.parameters()],  # if not any(nd in n for nd in no_decay)],
            #     "weight_decay": self.hparams.weight_decay,
            # },
            # {
            #     "params": [p for p in self.backbone_128.parameters()],  # if not any(nd in n for nd in no_decay)],
            #     "weight_decay": self.hparams.weight_decay,
            # },
            # {
            #     "params": [p for p in self.backbone_256.parameters()],  # if not any(nd in n for nd in no_decay)],
            #     "weight_decay": self.hparams.weight_decay,
            # },
            # {
            #    "params": [p for p in self.backbone_384.parameters()],  # if not any(nd in n for nd in no_decay)],
            #    "weight_decay": self.hparams.weight_decay,
            # },
            {
                "params": [p for p in self.backbone_512.parameters()],  # if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for p in self.reduce_dim_512.parameters()],  # if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for p in self.fc.parameters()],  # if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                      eps=self.hparams.adam_epsilon)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3], gamma=0.1)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": "val_loss"}
        return [optimizer], [scheduler]
