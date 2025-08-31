import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
import torch
from dataloader import CocoDetection, collate_fn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

coco_data_path = "mammo_1k/coco_1k"
print(coco_data_path)

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay, train_dataloader=None, val_dataloader=None):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=1,
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         self.train_dataloader = train_dataloader
         self.val_dataloader = val_dataloader


     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return self.train_dataloader

     def val_dataloader(self):
        return self.val_dataloader

PROCESSOR_PATH = "./processor.pt"
processor = DetrImageProcessor()
processor = torch.load(PROCESSOR_PATH)

coco_data_path = "../mammo_1k/coco_1k"
print(coco_data_path)


def train():
    
    
    train_dataset = CocoDetection(coco_data_folder=coco_data_path, processor=processor)
    val_dataset = CocoDetection(coco_data_folder=coco_data_path, processor=processor, train=False)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, num_workers=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
    batch = next(iter(train_dataloader))   
        
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
    
    
    checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',  # Metric to monitor for saving the best model
    dirpath='saved_models',  # Directory to save the checkpoints
    filename='best_model',  # Base name for the checkpoint files
    save_top_k=1,  # Save only the best model
    mode='min',  # Mode for comparison of monitored metric values
    )

    # Create the Trainer with the ModelCheckpoint callback
    trainer = Trainer(
        max_epochs=33, 
        gradient_clip_val=0.1, 
        callbacks=[checkpoint_callback]  # Add the checkpoint callback to the list of callbacks
    )

    # Train the model
    trainer.fit(model)
    # After training, load the best model checkpoint
    
    best_model_checkpoint_path = checkpoint_callback.best_model_path
    model_to_load = Detr.load_from_checkpoint(
        best_model_checkpoint_path, 
        lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    torch.save(model_to_load.state_dict(), "best_transformer_model.pt")