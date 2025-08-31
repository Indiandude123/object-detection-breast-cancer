import torch
import torchvision
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes
from dataloader import *
from sklearn.model_selection import train_test_split
from build_dataframe_coco import *
from torch.utils.data import DataLoader

MODEL_PATH = "./cnn_model.pt"



# Function to perform non-maximum suppression (NMS)
def apply_nms(boxes, scores, iou_threshold=0.5):
    # Convert boxes to (x1, y1, x2, y2) format expected by torchvision.ops.nms
    boxes_xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
    # Perform NMS
    keep = nms(boxes_xyxy, scores, iou_threshold)
    return keep

def convert_to_yolo_format(box, image_width, image_height):
    x_center = (box[0] + box[2]) / 2 / image_width
    y_center = (box[1] + box[3]) / 2 / image_height
    width = (box[2] - box[0]) / image_width
    height = (box[3] - box[1]) / image_height
    return x_center, y_center, width, height

def save_predictions_to_yolo_file(predictions, image_name, image_width, image_height, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Construct the full path for the output file
    # output_file = os.path.join(output_folder, f"{image_name}_preds.txt")
    output_file = f"{image_name}_preds.txt"
    with open(output_file, 'w') as f:
        for box, score in predictions:
            x_center, y_center, width, height = convert_to_yolo_format(box, image_width, image_height)
            line = f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
            f.write(line)



def test_and_save_predictions(validation_file_path, output_folder="./val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_load = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2
    in_features = model_to_load.roi_heads.box_predictor.cls_score.in_features
    model_to_load.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    if device == "gpu":
        model_to_load.load_state_dict(torch.load(MODEL_PATH))
        model_to_load.to(device)
    else:
        model_to_load.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_to_load.eval()
    
    image_path_arr = get_all_coco_images_path(validation_file_path, dataset_type="val")
    image_name_arr = get_all_coco_images_names(validation_file_path, dataset_type="val")
    df_coco_tumor, _, _ = read_coco_annotation_files(validation_file_path, dataset_type="val")
    df_coco_tumor_preprocessed = preprocess_coco_df(df_coco_tumor)
    
    val_data_loader = DataLoader(
        MamogramImagesDataset(df_coco_tumor_preprocessed, image_path_arr, image_name_arr, image_dims=(64, 224)),
        batch_size=1,
        collate_fn = custom_collate,
        shuffle=True,
        pin_memory = True if torch.cuda.is_available() else False)
        # Iterate over the validation dataloader
    for batch in val_data_loader:
        # Extract data from batch
        img = batch[0][0]
        image_name = batch[0][1]["image_name"]
        image_width = img.shape[2]
        image_height = img.shape[1]

        # Forward pass through the model
        with torch.no_grad():
            output = model_to_load([img.to(device)])
        
        # Apply NMS to filter redundant predictions
        keep = apply_nms(output[0]["boxes"], output[0]["scores"], iou_threshold=0.3)
        predictions = [(output[0]["boxes"][i], output[0]["scores"][i]) for i in keep]

        save_predictions_to_yolo_file(predictions, image_name, image_width, image_height, validation_file_path)
        
        
test_and_save_predictions(validation_file_path="../mammo_1k/coco_1k", output_folder="./val")