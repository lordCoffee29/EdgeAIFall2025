import numpy as np
import cv2

# Testing/debugging
def draw_boxes_no_labels(frame, bboxes, color=(0,0,255), thickness=2):
    """
    bboxes: list of (x1, y1, x2, y2) in pixel coords
    color: BGR; red=(0,0,255)
    """
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, thickness)
    return frame

# Crop the image
def crop_face(frame_bgr, det: Dict, margin: float=0.15, out_size=(224,224)):
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

    # margin around the face
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2.0, y1 + h/2.0
    w2, h2 = w*(1+margin), h*(1+margin)
    x1m = max(0, int(cx - w2/2)); y1m = max(0, int(cy - h2/2))
    x2m = min(W, int(cx + w2/2)); y2m = min(H, int(cy + h2/2))

    face = frame_bgr[y1m:y2m, x1m:x2m]
    if face.size == 0:
        return None, (x1m, y1m, x2m, y2m)
    face_resized = cv2.resize(face, out_size, interpolation=cv2.INTER_LINEAR)
    return face_resized, (x1m, y1m, x2m, y2m)




