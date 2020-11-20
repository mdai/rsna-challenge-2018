import os
import keras
import pydicom
from io import BytesIO
import tensorflow as tf
import numpy as np

from keras_retinanet import models
from keras import backend as K
from tensorflow import Graph, Session
from helper import read_image_dicom, resize_image, preprocess_image, histogram_equalize


class MDAIModel:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "../stage_2.h5")
        self.anchor_boxes = "0.25,0.33,0.5,0.75,1,1.33,2,3,4"
        self.score_threshold = 0.2
        self.nms_threshold = 0.1
        self.histogram_eq = True

    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []

        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            graph1 = Graph()
            with graph1.as_default():
                session1 = Session()
                with session1.as_default():
                    model = models.load_model(
                        self.model_path,
                        backbone_name="resnet101",
                        convert=True,
                        nms_threshold=self.nms_threshold,
                        anchors_ratios=[
                            float(item) for item in self.anchor_boxes.split(",")
                        ],
                    )
                    arr, ds = read_image_dicom(BytesIO(file["content"]))
                    if self.histogram_eq:
                        arr = histogram_equalize(arr[:, :, 0]) * 255
                        arr = np.stack((arr,) * 3, -1)
                    
                    del model

                    image = preprocess_image(arr.copy())
                    image, scale = resize_image(image)

                    if keras.backend.image_data_format() == "channels_first":
                        image = image.transpose((2, 0, 1))

                    # run network
                    boxes, scores, labels = model.predict_on_batch(
                        np.expand_dims(image, axis=0),
                    )[:3]

                    # correct boxes for image scale
                    boxes /= scale

                    # select indices which have a score above the threshold
                    indices = np.where(scores[0, :] > self.score_threshold)[0]

                    # select those scores
                    scores = scores[0][indices]

                    # find the order with which to sort the scores
                    scores_sort = np.argsort(-scores)[:100]

                    # select detections
                    image_boxes = boxes[0, indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[0, indices[scores_sort]]

                    image_detections = np.concatenate(
                        [
                            image_boxes,
                            np.expand_dims(image_scores, axis=1),
                            np.expand_dims(image_labels, axis=1),
                        ],
                        axis=1,
                    )

                    boxes[:, :, 2] -= boxes[:, :, 0]
                    boxes[:, :, 3] -= boxes[:, :, 1]
                    rsna_boxes = boxes[0, indices[scores_sort], :]

                    selection = np.where(image_scores > self.score_threshold)[0]

                    if len(selection) == 0:
                        output = {
                            "type": "NONE",
                            "study_uid": str(ds.StudyInstanceUID),
                            "series_uid": str(ds.SeriesInstanceUID),
                            "instance_uid": str(ds.SOPInstanceUID),
                            "frame_number": None,
                        }
                        outputs.append(output)
                    else:
                        for i in selection:
                            elem = rsna_boxes[i, :]
                            data = {
                                "x": int(elem[0]),
                                "y": int(elem[1]),
                                "width": int(elem[2]),
                                "height": int(elem[3]),
                            }
                            output = {
                                "type": "ANNOTATION",
                                "study_uid": str(ds.StudyInstanceUID),
                                "series_uid": str(ds.SeriesInstanceUID),
                                "instance_uid": str(ds.SOPInstanceUID),
                                "frame_number": None,
                                "class_index": 0,
                                "probability": 1.0,
                                "data": data,
                            }
                            outputs.append(output)

        return outputs

