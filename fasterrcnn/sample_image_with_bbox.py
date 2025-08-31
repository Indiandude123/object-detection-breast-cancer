import cv2
import os



def show_image_and_bbox(input_image_path, x1, y1, x2, y2, output_image_path):
  image = cv2.imread(input_image_path)
  color = (0, 255, 0)
  thickness = 2
  cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
  output_image = os.path.join(output_image_path, "image_with_bbox.jpg")
  cv2.imwrite(output_image, image)