import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_all_coco_images_path(coco_file_path, dataset_type="train"):
#   coco_file_path = os.path.join(data_folder_path, "coco_1k")

  coco_image_folder_arr = []
  for subdir in os.listdir(coco_file_path):
      image_folder_name = os.path.join(coco_file_path, subdir)
      if dataset_type in subdir:
        for image_file_name in os.listdir(image_folder_name):
          coco_image_folder_arr.append(os.path.join(image_folder_name, image_file_name))

  return coco_image_folder_arr


def get_all_coco_images_names(coco_file_path, dataset_type = "train"):
  coco_image_folder_name_arr = []
  for subdir in os.listdir(coco_file_path):
      image_folder_name = os.path.join(coco_file_path, subdir)
      if dataset_type in subdir:
        for image_file_name in os.listdir(image_folder_name):
          coco_image_folder_name_arr.append(image_file_name)

  return coco_image_folder_name_arr


class MamogramImagesDataset(Dataset):
  def __init__(self, df, all_coco_images_path_arr, all_coco_images_name_arr, image_dims, transforms=None):
    self.df = df
    self.unique_images_paths = all_coco_images_path_arr
    self.unique_images_names = all_coco_images_name_arr
    # self.indices = indices
    self.image_dims = image_dims
    self.transforms = transforms

  def __len__(self):
    return len(self.unique_images_paths)

  def __getitem__(self, idx):
    image_path_to_read = self.unique_images_paths[idx]
    image_name = self.unique_images_paths[idx]
    image = cv2.imread(image_path_to_read, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    height, width = image.shape[:2]
    image_resized = cv2.resize(image, self.image_dims)

    image_tensor = torch.from_numpy(image_resized)
    image_tensor = image_tensor.permute(2, 0, 1)
    boxes = self.df[self.df["image_path"] == image_path_to_read].drop(columns = ["from_dataset"]).iloc[:, 3:].to_numpy()

    scaled_bboxes = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        scaled_x1 = int(x1 * (self.image_dims[0] / width))
        scaled_y1 = int(y1 * (self.image_dims[1] / height))
        scaled_x2 = int(x2 * (self.image_dims[0] / width))
        scaled_y2 = int(y2 * (self.image_dims[1] / height))
        scaled_bboxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
    scaled_bboxes_array = np.array(scaled_bboxes)

    labels = torch.ones((scaled_bboxes_array.shape[0],), dtype=torch.int64)
    if len(scaled_bboxes_array) == 0:
      scaled_bboxes_array = torch.zeros((0, 4), dtype=torch.float32)
    else:
      scaled_bboxes_array = torch.as_tensor(scaled_bboxes_array, dtype=torch.float32)
    target = {}
    target["image_name"] = image_name
    target["boxes"] = scaled_bboxes_array
    target["labels"] = labels

    return image_tensor, target

def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def custom_collate(data):
  return data
