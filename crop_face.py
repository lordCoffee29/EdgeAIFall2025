import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Sequence, Union


class FaceCropper:
    """
    Utility class to crop faces and draw boxes.
    """

    def __init__(self, margin: float = 0.15, out_size: Tuple[int, int] = (224, 224)):
        self.margin = float(margin)
        self.out_size = (int(out_size[0]), int(out_size[1]))

    def draw_boxes_no_labels(self,
                             frame: 'np.ndarray',
                             bboxes: Sequence[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]],
                             color: Tuple[int, int, int] = (0, 0, 255),
                             thickness: int = 2) -> 'np.ndarray':
        """Draw rectangles on the provided frame.

        Args:
            frame: image in BGR (as used by OpenCV).
            bboxes: sequence of (x1, y1, x2, y2) in pixel coords.
            color: BGR color tuple.
            thickness: rectangle thickness.

        Returns:
            The modified frame (same array object).
        """
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return frame

    def crop_face(self,
                  frame_bgr: 'np.ndarray',
                  det: Dict,
                  margin: Optional[float] = None,
                  out_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional['np.ndarray'], Tuple[int, int, int, int]]:
        """Crop a face from an image using detection bbox and optional margin/out size.

        Args:
            frame_bgr: source image in BGR.
            det: dict-like with keys 'x1','y1','x2','y2' giving pixel coordinates.
            margin: optional override for margin (fractional, e.g., 0.15).
            out_size: optional override for output size (width,height).

        Returns:
            (face_resized, (x1m,y1m,x2m,y2m)) where face_resized is the cropped/resized face
            or None if the crop had zero area. Coordinates are clamped to image bounds.
        """
        if margin is None:
            margin = self.margin
        if out_size is None:
            out_size = self.out_size

        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # margin around the face
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        w2, h2 = w * (1 + margin), h * (1 + margin)
        x1m = max(0, int(cx - w2 / 2)); y1m = max(0, int(cy - h2 / 2))
        x2m = min(W, int(cx + w2 / 2)); y2m = min(H, int(cy + h2 / 2))

        face = frame_bgr[y1m:y2m, x1m:x2m]
        if face.size == 0:
            return None, (x1m, y1m, x2m, y2m)
        face_resized = cv2.resize(face, out_size, interpolation=cv2.INTER_LINEAR)
        return face_resized, (x1m, y1m, x2m, y2m)


# Backwards-compatible function wrappers
_default_cropper = FaceCropper()

def draw_boxes_no_labels(frame, bboxes, color=(0, 0, 255), thickness=2):
    """Backward-compatible function wrapper that uses FaceCropper.draw_boxes_no_labels."""
    return _default_cropper.draw_boxes_no_labels(frame, bboxes, color=color, thickness=thickness)


def crop_face(frame_bgr, det: Dict, margin: float = 0.15, out_size=(224, 224)):
    """Backward-compatible function wrapper that uses FaceCropper.crop_face.

    Note: the wrapper passes margin/out_size to the underlying cropper, so it's
    functionally equivalent to the previous implementation.
    """
    return _default_cropper.crop_face(frame_bgr, det, margin=margin, out_size=out_size)




