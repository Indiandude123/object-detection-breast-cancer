
import os 
import pandas as pd
import json



def get_image_info(image_obj_arr):
  image_ids_dict = {}
  image_height_dict = {}
  image_width_dict = {}
  for image_obj in image_obj_arr:
    image_ids_dict[image_obj["id"]] = image_obj["file_name"]
    image_height_dict[image_obj["file_name"]] = image_obj["height"]
    image_width_dict[image_obj["file_name"]] = image_obj["width"]
  return image_ids_dict, image_height_dict, image_width_dict

def get_annotations_info(annotation_obj_arr, image_ids_dict):
  image_name_arr =[]
  class_label_arr =[]
  x_arr =[]
  y_arr =[]
  width_arr =[]
  height_arr =[]
  for annotation_obj in annotation_obj_arr:
    image_id = annotation_obj["image_id"]
    image_name = image_ids_dict[image_id]
    image_name_arr.append(image_name)
    class_label_arr.append(annotation_obj["category_id"])
    bbox = annotation_obj["bbox"]
    x_arr.append(bbox[0])
    y_arr.append(bbox[1])
    width_arr.append(bbox[2])
    height_arr.append(bbox[3])
  return image_name_arr, class_label_arr, x_arr, y_arr, width_arr, height_arr


def read_coco_annotation_files(coco_file_path, dataset_type=None, images_folder_name=None):
  image_path_arr=[]
  if dataset_type=="train":
    images_folder_name="train2017"
  elif dataset_type=="val":
    images_folder_name="val2017"
  else:
    images_folder_name=images_folder_name
  images_folder_path=os.path.join(coco_file_path, images_folder_name)
  for subdir in os.listdir(coco_file_path):
    subdir_data_path = os.path.join(coco_file_path, subdir)
    # print(subdir)
    # print(subdir_data_path)
    if dataset_type in subdir:
      images_folder_path = subdir_data_path
      # print(images_folder_path)
    if "annotations" in subdir:
      for subsubdir in os.listdir(subdir_data_path):
        annotation_file_path = os.path.join(subdir_data_path, subsubdir)
        if dataset_type in subsubdir:
          # print(annotation_file_path)
          with open(annotation_file_path) as f:
            json_file = json.load(f)
          image_obj_arr = json_file["images"]
          image_ids_dict, image_height_dict, image_width_dict = get_image_info(image_obj_arr)
          annotation_obj_arr = json_file["annotations"]
          image_name_arr, class_label_arr, x_arr, y_arr, width_arr, height_arr = get_annotations_info(annotation_obj_arr, image_ids_dict)


          data = {
            'image_path' : [images_folder_path]*len(image_name_arr),
            'image_name': image_name_arr,
            'class_label': class_label_arr,
            'x1': x_arr,
            'y1': y_arr,
            'width': width_arr,
            'height': height_arr,
            'from_dataset': ["coco_1k"]*len(image_name_arr)
          }
          df = pd.DataFrame(data)
          df['image_path'] = df.apply(lambda x: os.path.join(x["image_path"], x["image_name"]), axis=1)
  return df, image_height_dict, image_width_dict


def preprocess_coco_df(df_coco_tumor):
  df_coco_tumor["x2"] = df_coco_tumor["x1"] + df_coco_tumor["width"]
  df_coco_tumor["y2"] = df_coco_tumor["y1"] + df_coco_tumor["height"]

  df_coco_tumor_preprocessed = df_coco_tumor.drop(columns = ["width", "height"], inplace=False)
  return df_coco_tumor_preprocessed