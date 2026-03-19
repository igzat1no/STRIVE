import numpy as np
import supervision as sv
from supervision.geometry.core import Position


def visualize_mask(img, boxes, confidences=None, class_names=None, masks=None):
    if confidences is None and class_names is None:
        labels = None
    elif confidences is None and class_names is not None:
        labels = class_names
    elif confidences is not None and class_names is None:
        labels = [f"{confidence:.2f}" for confidence in confidences]
    else:
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]
    # input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    input_boxes = boxes.numpy().astype(int)

    if masks is None and class_names is None:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
        )
    elif masks is None and class_names is not None:
        class_ids = np.array(list(range(len(class_names))))
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            class_id=class_ids)
    elif masks is not None and class_names is None:
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.numpy().astype(bool),  # (n, h, w)
        )
    else:
        class_ids = np.array(list(range(len(class_names))))
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.numpy().astype(bool),  # (n, h, w)
            class_id=class_ids)

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    if labels is not None:
        label_annotator = sv.LabelAnnotator(text_position=Position.TOP_LEFT)
        annotated_frame = label_annotator.annotate(scene=annotated_frame,
                                                   detections=detections,
                                                   labels=labels)

    if masks is not None:
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame,
                                                  detections=detections)

    return annotated_frame
