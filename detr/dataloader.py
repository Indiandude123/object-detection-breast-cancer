import torchvision
import os
import torch
from transformers import DetrImageProcessor


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, coco_data_folder=None, processor=None, train=True, data_folder=None):
        if data_folder == None:
            annotation_folder = os.path.join(coco_data_folder, "annotations")
            ann_file = os.path.join(annotation_folder, "instances_train2017.json" if train else "instances_val2017.json")
            img_folder = os.path.join(coco_data_folder, "train2017" if train else "val2017")
            super(CocoDetection, self).__init__(img_folder, ann_file)
            self.processor = processor
        else:
            annotation_folder = os.path.join(data_folder, "annotations")
            ann_file = os.path.join(annotation_folder, "image_info_test-dev2017.json")
            
            img_folder = os.path.join(data_folder, "test")
            # print(img_folder)
            super(CocoDetection, self).__init__(img_folder, ann_file)
            # super(CocoDetection, self).__init__(img_folder)
            self.processor = processor
            

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
PROCESSOR_PATH = "./processor.pt"
processor = DetrImageProcessor()
processor = torch.load(PROCESSOR_PATH)


def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch