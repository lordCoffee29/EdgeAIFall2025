"""YOLOX face detection helper.

This module provides a lightweight wrapper around the repository's YOLOX/ONNX
inference helpers (see `detect_face.py`) to produce face bounding boxes from
an image. It returns boxes as dictionaries compatible with the `FaceCropper` in
`crop_face.py` (keys: 'x1','y1','x2','y2','score').

Example:
    from get_crop_points import YOLOXFaceDetector

    det = YOLOXFaceDetector(base="/path/to/model_folder")
    boxes = det.detect(frame)  # list of {'x1','y1','x2','y2','score'}

"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import onnxruntime as ort

try:
    import crop_face
except Exception:
    crop_face = None

# Reuse helpers from detect_face.py in the repo
try:
    import detect_face
except Exception:
    detect_face = None


class YOLOXFaceDetector:
    """Loads a YOLOX ONNX model (using same folder layout as detect_face.py) and
    provides a .detect(frame) method that returns face bounding boxes.

    Args:
        base: path to the model package root (contains artifacts/model.onnx and optional config.yaml)
        providers: list of ONNXRuntime providers (defaults to ['CPUExecutionProvider'])
        input_size: optional override for model input size (square)
        score_thr: optional override for detection score threshold
        iou_thr: optional NMS IoU threshold
        pad_center: whether to use centered letterbox when preprocessing
    """

    def __init__(self,
                 base: str,
                 providers: Optional[List] = None,
                 input_size: int = 0,
                 score_thr: Optional[float] = None,
                 iou_thr: float = 0.45,
                 pad_center: bool = False):
        self.base = os.path.abspath(base)
        self.providers = providers or ["CPUExecutionProvider"]
        self.input_size_override = int(input_size)
        self.score_override = score_thr
        self.iou = float(iou_thr)
        self.pad_center = bool(pad_center)

        # find ONNX path similar to detect_face.py
        # First try the standard paths
        artifacts = os.path.join(self.base, "artifacts")
        model_dir = os.path.join(self.base, "model")
        
        # List of paths to try
        possible_paths = [
            os.path.join(artifacts, "model.onnx"),
            os.path.join(model_dir, "model.onnx"),
            # Try to find any .onnx file in model directory
            *[os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.onnx')]
        ]
        
        onnx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                onnx_path = path
                break
                
        if onnx_path is None:
            raise FileNotFoundError(
                f"No ONNX model found in {artifacts} or {model_dir}")

        # load config via detect_face helper if available
        cfg = {}
        if detect_face is not None:
            try:
                cfg = detect_face.load_cfg(self.base)
            except Exception:
                cfg = {}

        self.size = detect_face.get_input_size(cfg, self.input_size_override) if detect_face else (self.input_size_override or 640)
        self.score_thr = detect_face.get_score_thr(cfg, self.score_override) if detect_face else (self.score_override or 0.35)
        if detect_face:
            lb_on, pad_mode, pad_color, reverse = detect_face.get_letterbox_mode(cfg, self.pad_center)
            self.letterbox_on = lb_on
            self.pad_mode = pad_mode
            self.pad_color = pad_color
            self.reverse = reverse
        else:
            self.letterbox_on = True
            self.pad_mode = "corner"
            self.pad_color = (114,114,114)
            self.reverse = True

        # create ONNXRuntime session
        # Accept either provider names or (provider, options) pairs like detect_face does
        ort_providers = []
        for p in self.providers:
            if isinstance(p, (list, tuple)):
                ort_providers.append((p[0], p[1] if len(p) > 1 else {}))
            else:
                ort_providers.append(p)

        self.sess = ort.InferenceSession(onnx_path, providers=ort_providers)
        self.in_name = self.sess.get_inputs()[0].name
        # Debug: print model output metadata
        # silence model output metadata in normal runs; keep exceptions visible
        try:
            _ = [(o.name, o.shape, o.type) for o in self.sess.get_outputs()]
        except Exception:
            pass

    def detect(self, frame: "np.ndarray", return_crops: bool = False, return_all: bool = False):
        """Run detection on a BGR image and return list of boxes.

        Args:
            frame: BGR image (numpy array)
            return_crops: if True, also return cropped face images as a list.
            return_all: if True, return all postprocessed boxes (not only the top one).

        Returns:
            If return_crops is False: list of detection dicts each with keys 'x1','y1','x2','y2'.
            If return_crops is True: tuple (detections, crops) where crops is a list of
            cropped BGR face images (or None for failed crops).
        """
        if detect_face is None:
            raise RuntimeError("detect_face helpers not available in the repo; cannot preprocess/postprocess reliably")

        # Get normalized float32 input and metadata
        inp, meta = detect_face.preprocess_bgr(frame, self.size,
                                               reverse_channels=self.reverse,
                                               letterbox_on=self.letterbox_on,
                                               pad_mode=self.pad_mode,
                                               pad_color=self.pad_color)
        
        # YOLOX takes in uint8 inputs
        inp_uint8 = (inp * 255).astype(np.uint8)
        outs = self.sess.run(None, {self.in_name: inp_uint8})
        boxes = detect_face.postprocess(outs, meta, score_thr=self.score_thr, iou_thr=self.iou, letterbox_on=self.letterbox_on)

        # postprocess returns list of (x1,y1,x2,y2,score) ints.
        results = []
        crops = []

        # choose which boxes to keep: either all or the single top-scoring one
        # TO-DO: only pass in the highest scoring box into the crop list
        chosen_boxes = boxes if return_all else ([max(boxes, key=lambda b: b[4])] if boxes else [])

        for (x1,y1,x2,y2,score) in chosen_boxes:
            det = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            results.append(det)
            # Emit concise detection line without confidence
            print(f"Detected face at (x1={det['x1']}, y1={det['y1']}, x2={det['x2']}, y2={det['y2']})")
            if return_crops:
                if crop_face is None:
                    crops.append(None)
                else:
                    face_img, coords = crop_face._default_cropper.crop_face(frame, det)
                    crops.append(face_img)

        if return_crops:
            return results, crops
        return results


_detector_singleton: Optional[YOLOXFaceDetector] = None

def get_crop_points_from_image(frame: "np.ndarray", base: str, **kwargs) -> List[Dict]:
    """Simple helper: load (or reuse) a detector for `base` and return boxes for `frame`.

    Args:
        frame: BGR image (as from cv2.imread or camera)
        base: model package folder (same as detect_face -- contains artifacts/model.onnx)
        kwargs: forwarded to YOLOXFaceDetector constructor (e.g., score_thr, input_size)
    """
    global _detector_singleton
    if _detector_singleton is None or _detector_singleton.base != os.path.abspath(base):
        _detector_singleton = YOLOXFaceDetector(base=base, **kwargs)
    return _detector_singleton.detect(frame)


if __name__ == "__main__":
    print("Module provides YOLOXFaceDetector and get_crop_points_from_image(frame, base, **kwargs)")
