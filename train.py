import torch
from pytorch_lightning import Trainer, seed_everything

from dataset import DataModule
from model import MayoModel

AVAIL_GPUS = min(1, torch.cuda.device_count())

seed_everything(3)

dm = DataModule(train_batch_size=1, eval_batch_size=1, train_or_test='train')
print('dm is created')

model = MayoModel(
    eval_splits=dm.eval_splits,
    # weight_decay=1e-3,
    train_batch_size=dm.train_batch_size,
    eval_batch_size=dm.eval_batch_size,
    learning_rate=1e-3

)

print('model is created')
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=-1
)

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs", name='512')


trainer = Trainer(max_epochs=100,
                  gpus=AVAIL_GPUS,
                  check_val_every_n_epoch=1,
                  # auto_scale_batch_size=True,
                  callbacks=[checkpoint_callback,
                             StochasticWeightAveraging(0.99)
                             ],
                  logger=logger,
                  accumulate_grad_batches=16,

                  # accelerator="gpu",
                  # devices=4,
                  # strategy=DDPStrategy(find_unused_parameters=False),
                  # replace_sampler_ddp=False,
                  )
print('trainer is created')
trainer.fit(model, datamodule=dm)
