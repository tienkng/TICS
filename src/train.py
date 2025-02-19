import os

import torch.utils
import torch.utils.data
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging

import torch
import torch.nn.functional as F
from typing import Any, Optional

from torchmetrics import MeanMetric
from torchmetrics.classification import F1Score

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from modeling.model import build_model
from data.dataset import TicDataset
from arguments import get_args
from src.utils.config import read_config_from_file

logger = logging.getLogger(__name__)



class LitTic(LightningModule):
    def __init__(self, model=None, lr:float=0.01):
        super().__init__()    
        self.model = model
        self.lr = lr
        
        # define val metric
        self.mean_valid_loss = MeanMetric()
        self.val_f1 = F1Score(task="multiclass", num_classes=7)

    def forward(self, x):        
        logits = self.model(x)

        return logits
        
    def training_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        logits = self(inputs)
        loss = F.cross_entropy(logits, target.long())
        
        self.log("train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True,  sync_dist=self.sync_dist)
  
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        with torch.no_grad():
            logits = self(inputs)

        # Loss
        loss = F.cross_entropy(logits, target.long()) 
        self.mean_valid_loss.update(loss, weight=inputs.shape[0])
        
        # Metric
        pred_classes = torch.argmax(pred_classes, dim=1)
        self.val_f1.update(pred_classes, target)
        
        return loss
    
    def on_validation_epoch_end(self):
        # compute metrics
        val_f1 = self.val_f1.compute()
        
        # log metrics
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True, sync_dist=self.sync_dist, logger=True)
        self.log("valid/f1_score", val_f1, prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        # reset all metrics
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        inputs, target = batch[0], batch[1]
        with torch.no_grad():
            pred_age, pred_gender, pred_emotion = self(inputs) 
            pred_age, pred_gender, pred_emotion = pred_age.squeeze(), pred_gender.squeeze(), pred_emotion.squeeze()
        
        # Metric
        pred_emotion = torch.argmax(pred_emotion, dim=1)
        self.val_f1.update(pred_emotion, target)
    
    def on_test_epoch_end(self) -> None:
        val_score = self.val_f1.compute()
        self.log("test/f1_score", val_score, sync_dist=self.sync_dist, logger=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer =  torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-4)
        
        return [optimizer]

    def save_checkpoint(self, filepath, weights_only:bool=False, storage_options:Optional[Any]=None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")


def main(args):
    # setup config
    wandb_logger = WandbLogger(project=args.project_name, log_model="all")
    
    model_config = read_config_from_file(args.model_config)
    model = build_model(model_config)
    
    # create Model wrapper
    litmodel = LitTic(model, lr=args.lr)
    
    # create dataset
    if args.do_train:
        training_dataset = TicDataset(
            data_path=args.training_dir,
            img_size=model_config.model_kwargs.image_size,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            start_sec=args.start_sec,
            frames_per_second=args.frames_per_second,
            seq_len=model_config.model_kwargs.seq_len,
            training_mode=True
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        
    if args.do_eval:
        val_dataset = TicDataset(
            data_path=args.validation_dir,
            img_size=model_config.model_kwargs.image_size,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            start_sec=args.start_sec,
            frames_per_second=args.frames_per_second,
            seq_len=model_config.model_kwargs.seq_len,
            training_mode=False
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        
        
    # create callback functions
    model_checkpoint = ModelCheckpoint(save_top_k=5,
                        monitor="valid/loss",
                        mode="min", dirpath=args.output_path,
                        filename="sample-{epoch:02d}",
                        save_weights_only=True)
    
    dist = True if len(args.device) > 1 else False

    # create Trainer
    trainer = Trainer(
        max_epochs=args.epochs, 
        accelerator=args.accelerator, 
        devices=args.device, 
        callbacks=[model_checkpoint], 
        strategy='ddp_find_unused_parameters_true' if dist else 'auto',
        log_every_n_steps=args.log_steps,
        logger=wandb_logger,
    )
    
    if args.do_train:
        logger.info("*** Start training ***")
        trainer.fit(
            model=litmodel, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader if args.do_eval else None
        )
        
        # Saves only on the main process    
        saved_ckpt_path = f'{saved_ckpt_path}/checkpoint'
        os.makedirs(saved_ckpt_path, exist_ok=True)
        saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
        trainer.save_checkpoint(saved_ckpt_path)
        
    if args.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(litmodel, dataloaders=val_dataloader, ckpt_path="best")
        
        
if __name__ == '__main__':
    opt = get_args()
    
    print("\nHyperparameters\n", opt, "\n")
    
    # trainer
    logger.info('*** Training mode ***')
    main(opt)
    