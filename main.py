import streamlit as st
import torch, torchvision
# import req
import numpy as np
import os, json, random, cv2
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import Metadata
from PIL import Image, ImageOps
''''''

@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.

    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.

    """
    # Import some common detectron2 utilities
    NUM_CLASSES = 7
    # Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    # Custom Trained Model
    cfg.MODEL.WEIGHTS = os.path.join("model_final.pth")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 128

    # minimum image size for the train set
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    # maximum image size for the train set
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # minimum image size for the test set
    cfg.INPUT.MIN_SIZE_TEST = 800
    # maximum image size for the test set
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Predictor
    predictor = DefaultPredictor(cfg)

    return cfg, predictor


@st.cache
def inference(predictor, img):
    return predictor(img)

# Detectron2 Setup Logger
setup_logger()


# Detectron2 Metadata
my_metadata = Metadata()
my_metadata.set(thing_classes=['metals_and_plastics', 'other', 'non-recyclable', 'glass', 'paper', 'bio', 'unknown'])
cats = ['metals_and_plastics', 'other', 'non-recyclable', 'glass', 'paper', 'bio', 'unknown']

# Streamlit Title
st.title('Waste Detector:')
st.caption('Using Detectron2 (BBox+Segmentation)')

st.write('\n')

# st.write('Test image')
# im = cv2.imread("assets/test.JPG")
# # showing image
# st.image('assets/test.JPG')



###
# Upload images
st.write('Upload image')
upload = st.file_uploader("Please upload an image", type=["jpg","png", "jpeg", "heif"])
if upload is None:
    st.text("Please upload an image")
else:
    st.write('Image Uploaded:')
    image = Image.open(upload)
    st.image(image, use_column_width=True)
    img_array = np.array(image)
    cfg, predictor = initialization()

    outputs = predictor(img_array)

    st.title('Prediction Outputs:')
    st.write('Pred Classes: ',outputs["instances"].pred_classes)
    st.write('Categories: ',cats)
    # st.write('Pred Boxes: ',outputs["instances"].pred_boxes)
    st.write('Scores: ', outputs["instances"].scores)

    st.write('Using Vizualizer to draw the predictions on Image')

    v = Visualizer(img_array[:, :, ::-1], metadata=my_metadata, instance_mode=ColorMode.IMAGE_BW, # removes the colors of unsegmented pixels
                    scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    st.image(out.get_image()[:, :, ::-1])
''''''
