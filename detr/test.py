import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from dataloader import CocoDetection, collate_fn
from train import Detr
from tqdm import tqdm
import os

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def convert_coco_to_yolo(predictions, image_width, image_height):
    yolo_predictions = []
    for prediction in predictions:
        # Extract prediction information
        category_id = prediction['category_id']
        bbox = prediction['bbox']
        score = prediction['score']
        
        # Calculate center coordinates and dimensions in YOLO format
        x_center = (bbox[0] + bbox[2]) / (2 * image_width)
        y_center = (bbox[1] + bbox[3]) / (2 * image_height)
        bbox_width = (bbox[2] - bbox[0]) / image_width
        bbox_height = (bbox[3] - bbox[1]) / image_height
        
        # Append YOLO format prediction to the list
        yolo_predictions.append([x_center, y_center, bbox_width, bbox_height, score])
    
    return yolo_predictions

def save_yolo_predictions(predictions, output_file, directory):
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Construct the full path for the output file
    output_path = os.path.join(directory, output_file)
    # Write predictions to the file
    with open(output_path, 'w') as f:
        for prediction in predictions:
            line = ' '.join(map(str, prediction)) + '\n'
            f.write(line)


PROCESSOR_PATH = "./processor.pt"
processor_to_load = DetrImageProcessor()
processor_to_load = torch.load(PROCESSOR_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./best_transformer_model.pt"
model_to_load = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
if device == "gpu":
    model_to_load.load_state_dict(torch.load(MODEL_PATH))
    model_to_load.to(device)
else:
    model_to_load.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_to_load.eval()
# model_to_load.to(device)


def val(coco_data_path, save_dir="./val"):
    val_dataset = CocoDetection(coco_data_folder=coco_data_path, train=False, processor=processor_to_load)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = model_to_load(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

        results = processor_to_load.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)
        # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        
    #     image = val_dataset.coco.loadImgs(image_id)[0]
        predictions = prepare_for_coco_detection(predictions)
        # print(predictions)
        if len(predictions):
            image_id = predictions[0]["image_id"]
            image = val_dataset.coco.loadImgs(image_id)[0]
            image_name = image["file_name"]
            output_file = image_name[:-4]+ "_preds.txt"
            yolo_format_preds = convert_coco_to_yolo(predictions, image["width"], image["height"])
            output_directory = os.path.join(save_dir, "predictions")
            save_yolo_predictions(yolo_format_preds, output_file, output_directory)



def test(data_path, save_dir="./test"):
    val_dataset = CocoDetection(processor=processor_to_load, data_folder=data_path)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
    for idx, batch in enumerate(tqdm(val_dataloader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = model_to_load(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)

        results = processor_to_load.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)
        # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        
    #     image = val_dataset.coco.loadImgs(image_id)[0]
        predictions = prepare_for_coco_detection(predictions)
        # print(predictions)
        if len(predictions):
            image_id = predictions[0]["image_id"]
            image = val_dataset.coco.loadImgs(image_id)[0]
            image_name = image["file_name"]
            output_file = image_name[:-4]+ "_preds.txt"
            yolo_format_preds = convert_coco_to_yolo(predictions, image["width"], image["height"])
            output_directory = os.path.join(save_dir, "predictions")
            save_yolo_predictions(yolo_format_preds, output_file, output_directory)
 
            
# coco_data_path = "../mammo_1k/coco_1k"
# val(coco_data_path)

data_path = "../test_images"
test(data_path)