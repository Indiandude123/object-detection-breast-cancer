import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataloader import *
from sklearn.model_selection import train_test_split
from build_dataframe_coco import *

IMAGE_DIMS = (64, 224)
MODEL_PATH = "./cnn_model.pt"


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0



def train(coco_file_path, fold_type="train"):
    if fold_type == "train":
        
        image_path_arr = get_all_coco_images_path(coco_file_path, fold_type)
        image_name_arr = get_all_coco_images_names(coco_file_path, fold_type)
        df_coco_tumor, _, _ = read_coco_annotation_files(coco_file_path, fold_type)
        df_coco_tumor_preprocessed = preprocess_coco_df(df_coco_tumor)
        # train_inds, valid_inds = train_test_split(range(len(image_folder_arr)), test_size=TEST_SIZE)
        train_data_loader = DataLoader(
            MamogramImagesDataset(df_coco_tumor_preprocessed, image_path_arr, image_name_arr, IMAGE_DIMS),
            batch_size=4,
            collate_fn = custom_collate,
            shuffle=True,
            pin_memory = True if torch.cuda.is_available() else False)

        # val_data_loader = DataLoader(
        #     MamogramImagesDataset(df_coco_tumor_preprocessed, image_folder_arr, valid_inds, IMAGE_DIMS),
        #     batch_size=8,
        #     collate_fn = custom_collate,
        #     shuffle=True,
        #     pin_memory = True if torch.cuda.is_available() else False)
        
        
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = None
        num_epochs = 2
        
        loss_hist = Averager()
        itr = 1

        for epoch in range(num_epochs):
            loss_hist.reset()
            for data in train_data_loader:
                images = []
                targets = []
                for d in data:
                    images.append(d[0].to(device))
                    target_obj = {}
                    target_obj["boxes"] = d[1]["boxes"].to(device)
                    target_obj["labels"] = d[1]["labels"].to(device)
                    targets.append(target_obj)

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if itr % 50 == 0:
                    print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1

            if lr_scheduler is not None:
                lr_scheduler.step()

            print(f"Epoch #{epoch} loss: {loss_hist.value}")
            
        torch.save(model.state_dict(), MODEL_PATH)
        
        
            