from PIL import Image
import io
import pandas as pd
import numpy as np

from typing import Optional

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import CocoPrediction
# Initialize the models

model_sample_detect = YOLO("./models/sample_model/best.pt")
model_wall_detect = YOLO("./models/sample_model/wall_detect.pt")

detection_model = AutoDetectionModel.from_pretrained(
    model_type= 'yolov8',
    model_path= r'./models/sample_model/best.pt',
    confidence_threshold = 0.75,
    device="cuda:0", # or 'cuda:0'
)

def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    **Args:**
        - **binary_image (bytes):** The binary representation of the image
    
    **Returns:**
        - **PIL.Image:** The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.51, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        iou = 0.3                       
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


# def get_model_segment(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.25, augment: bool = False) -> pd.DataFrame:
#     """
#     Get the predictions of a model on an input image.
    
#     Args:
#         model (YOLO): The trained YOLO model.
#         input_image (Image): The image on which the model will make predictions.
#         save (bool, optional): Whether to save the image with the predictions. Defaults to False.
#         image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
#         conf (float, optional): The confidence threshold for the predictions. Defaults to 0.25.
#         augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
#     Returns:
#         pd.DataFrame: A DataFrame containing the predictions.
#     """
#     # Make predictions
#     predictions = model.predict(
#                         imgsz=image_size, 
#                         source=input_image, 
#                         conf=conf,
#                         save=save, 
#                         augment=augment,
#                         flipud= 0.0,
#                         fliplr= 0.0,
#                         mosaic = 0.0,
#                         )
    
#     # Transform predictions to pandas dataframe
#     predictions = transform_predict_to_df(predictions, model.model.names)
#     return predictions


################################# BBOX Func #####################################

# def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
#     """
#     add a bounding box on the image

#     Args:
#     image (Image): input image
#     predict (pd.DataFrame): predict from model

#     Returns:
#     Image: image whis bboxs
#     """
#     # Create an annotator object
#     # print(image)
#     annotator = Annotator(np.array(image), line_width=None, font_size=10, pil=True)

#     # sort predict by xmin value
#     predict = predict.sort_values(by=['xmin'], ascending=True)
#     # iterate over the rows of predict dataframe
#     for i, row in predict.iterrows():
#         # create the text to be displayed on image
#         text = f"{row['name']}: {int(row['confidence']*100)}%"
#         # get the bounding box coordinates
#         bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
#         # add the bounding box and text on the image
#         annotator.box_label(bbox
#                             ,text
#                             ,color=colors(row['class'], True))
#     # convert the annotated image to PIL image
#     return Image.fromarray(annotator.result())

# def add_wall_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
#     """
#     add a bounding box on the image

#     Args:
#     image (Image): input image
#     predict (pd.DataFrame): predict from model

#     Returns:
#     Image: image whis bboxs
#     """
#     # Create an annotator object
#     annotator = Annotator(np.array(image), line_width=2)

#     # sort predict by xmin value
#     predict = predict.sort_values(by=['xmin'], ascending=True)
#     # iterate over the rows of predict dataframe
#     for i, row in predict.iterrows():
#         # create the text to be displayed on image
#         text = f"{row['name']}: {int(row['confidence']*100)}%"
#         # get the bounding box coordinates
#         bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
#         # add the bounding box and text on the image
#         annotator.box_label(bbox,
#                             # text,
#                             # color=(255,0,0)
#                             color=colors(row['class'], False)
#                             )
#     # convert the annotated image to PIL image
#     return Image.fromarray(annotator.result())
################################# Models #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_sample_detect,
        input_image=input_image,
        save=False,
        augment=False,
        conf=0.51,
    )
    return predict

def toxyxy(self):
    coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_xyxy(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=None,
            )
    return coco_prediction

def to_xyxy_annotations(self):
    coco_annotation_list = []
    for object_prediction in self.object_prediction_list:
        coco_annotation_list.append(toxyxy(object_prediction).json)
    coco_df = pd.DataFrame(coco_annotation_list)
    coco_df[['xmin', 'ymin', 'xmax', 'ymax']] = pd.DataFrame(coco_df['bbox'].tolist())
    coco_df = coco_df.drop(columns=['bbox','segmentation','image_id','iscrowd','area'])
    coco_df.rename(columns={"score": "confidence", "category_name": "name", "category_id": "class"},inplace=True)
    return coco_df

def sliced_detect_object_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    result = get_sliced_prediction(
        input_image,
        detection_model,
        slice_height = input_image.size[1],
        slice_width = input_image.size[0],
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )
    return to_xyxy_annotations(result)

# def sliced_detect_wall_model(input_image: Image) -> pd.DataFrame:
#     """
#     Predict from sample_model.
#     Base on YoloV8

#     Args:
#         input_image (Image): The input image.

#     Returns:
#         pd.DataFrame: DataFrame containing the object location.
#     """
#     predict = get_model_predict(
#         model=model_wall_detect,
#         input_image=input_image,
#         save=False,
#         image_size=640,
#         augment=False,
#         conf=0.25,
#     )
#     return predict

def detect_wall_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_wall_detect,
        input_image=input_image,
        save=False,
        augment=False,
        conf=0.25,
    )
    return predict

# def segment_sample_model(input_image: Image) -> pd.DataFrame:
#     """
#     Predict from sample_model.
#     Base on YoloV8

#     Args:
#         input_image (Image): The input image.

#     Returns:
#         pd.Dataframe: Dataframe containing the object location and segmentation.
#     """
#     predict = get_model_segment(
#         model=model_sample_segment,
#         input_image=input_image,
#         save=True,
#         image_size=640,
#         augment=False,
#         conf=0.25,
#     )
#     return predict