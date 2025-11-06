#!/usr/bin/env python3
import os, time, argparse, json
import numpy as np
import cv2

# --- YAML is optional; we fall back to defaults if it's missing ---
try:
    import yaml
except Exception:
    yaml = None

import onnxruntime as ort

def parse_args():
    ap = argparse.ArgumentParser("TDA4VM YOLOX Face (TIDL) – red boxes")
    ap.add_argument("--base", required=True,
                    help="Model folder (contains artifacts/, config.yaml, etc.), e.g. /opt/edgeai/model_zoo/TestModel")
    ap.add_argument("--video", default="", help="Path to input video; omit to use a camera")
    ap.add_argument("--camera", type=int, default=0, help="Camera index if not using --video")
    ap.add_argument("--save", default="", help="Optional path to save annotated MP4 (e.g., /opt/videos/out.mp4)")
    ap.add_argument("--score", type=float, default=None, help="Override score threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    ap.add_argument("--input-size", type=int, default=0, help="Override input square size (e.g., 640 or 320)")
    ap.add_argument("--rgb", action="store_true", help="Force RGB input (override config reverse_channels)")
    ap.add_argument("--pad-center", action="store_true", help="Use centered letterbox instead of corner")
    return ap.parse_args()

# -------------- helpers --------------

def load_cfg(base):
    cfg = {}
    if yaml is None:
        return cfg
    ypath = os.path.join(base, "config.yaml")
    if os.path.exists(ypath):
        try:
            with open(ypath, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    return cfg

def get_input_size(cfg, override):
    if override > 0:
        return override
    pp = cfg.get("preprocess", {})
    # prefer crop then resize (TI YAML often specifies both)
    return int(pp.get("crop", pp.get("resize", 640)))

def get_score_thr(cfg, override):
    if override is not None:
        return float(override)
    post = cfg.get("postprocess", {})
    return float(post.get("detection_threshold", 0.35))

def get_letterbox_mode(cfg, force_center):
    pp = cfg.get("preprocess", {})
    pad = pp.get("resize_with_pad", [True, "corner"])
    if isinstance(pad, list):
        lb = bool(pad[0]); mode = pad[1] if len(pad) > 1 else "corner"
    else:
        lb = bool(pad); mode = "corner"
    if force_center:
        mode = "center"
    pad_color = tuple(pp.get("pad_color", [114,114,114]))
    reverse = bool(pp.get("reverse_channels", True))
    return lb, mode, pad_color, reverse

def letterbox(img, new_size, color=(114,114,114), mode="corner"):
    h, w = img.shape[:2]
    r = min(new_size / w, new_size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=resized.dtype)
    if mode == "center":
        dx = (new_size - nw) // 2
        dy = (new_size - nh) // 2
    else:  # "corner" (top-left)
        dx, dy = 0, 0
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, r, dx, dy, (w, h)

def preprocess_bgr(frame, size, reverse_channels=True, letterbox_on=True, pad_mode="corner", pad_color=(114,114,114)):
    if letterbox_on:
        lb, r, dx, dy, orig_wh = letterbox(frame, size, color=pad_color, mode=pad_mode)
        img = lb
    else:
        img = cv2.resize(frame, (size, size))
        r, dx, dy, orig_wh = None, 0, 0, (frame.shape[1], frame.shape[0])

    img = img.astype(np.float32) * (1.0/255.0)
    if reverse_channels:
        img = img[..., ::-1]          # BGR->RGB if model expects RGB
    img = np.transpose(img, (2,0,1))  # HWC->CHW
    return img[np.newaxis, ...], (r, dx, dy, orig_wh)

def nms_iou(dets, iou_thr=0.45):
    if not dets:
        return []
    dets = np.array(dets, dtype=np.float32)  # [x1,y1,x2,y2,score]
    x1,y1,x2,y2,s = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    order = s.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0, xx2-xx1); ih = np.maximum(0, yy2-yy1)
        inter = iw*ih
        area_i = (x2[i]-x1[i])*(y2[i]-y1[i])
        area_j = (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou < iou_thr]
    return dets[keep].tolist()

def postprocess(outputs, meta, score_thr=0.35, iou_thr=0.45, letterbox_on=True):
    """
    Supports two common TI/YOLOX output formats:
      A) [dets=(N,5), labels=(N,)] where dets=[x1,y1,x2,y2,score] in letterbox space
      B) [raw=(1,M,6)] where each row is [x1,y1,x2,y2,score,cls]
    """
    (r, dx, dy, (ow, oh)) = meta
    boxes = []

    o0 = outputs[0]
    if o0.ndim == 2 and o0.shape[1] == 5:
        # Format A: dets + labels
        dets = o0
        for x1,y1,x2,y2,score in dets:
            score = float(score)
            if score < score_thr: continue
            x1,y1,x2,y2 = map(float, (x1,y1,x2,y2))
            if letterbox_on:
                x1 = (x1 - dx) / (r + 1e-12)
                y1 = (y1 - dy) / (r + 1e-12)
                x2 = (x2 - dx) / (r + 1e-12)
                y2 = (y2 - dy) / (r + 1e-12)
            # clamp
            x1 = max(0, min(ow-1, x1)); y1 = max(0, min(oh-1, y1))
            x2 = max(0, min(ow-1, x2)); y2 = max(0, min(oh-1, y2))
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1,y1,x2,y2,score])

    elif o0.ndim == 3 and o0.shape[2] >= 6:
        # Format B: (1,M,6+)  [x1,y1,x2,y2,score,cls]
        raw = o0[0]
        for row in raw:
            x1,y1,x2,y2,score = map(float, row[:5])
            if score < score_thr: continue
            if letterbox_on:
                x1 = (x1 - dx) / (r + 1e-12)
                y1 = (y1 - dy) / (r + 1e-12)
                x2 = (x2 - dx) / (r + 1e-12)
                y2 = (y2 - dy) / (r + 1e-12)
            x1 = max(0, min(ow-1, x1)); y1 = max(0, min(oh-1, y1))
            x2 = max(0, min(ow-1, x2)); y2 = max(0, min(oh-1, y2))
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1,y1,x2,y2,score])

    else:
        # Unknown format → print and exit gracefully
        print("Unexpected output shapes:", [o.shape for o in outputs])
        return []

    boxes = nms_iou(boxes, iou_thr)
    # ints for drawing
    return [(int(b[0]),int(b[1]),int(b[2]),int(b[3]),float(b[4])) for b in boxes]

def draw_boxes(frame, boxes, color=(0,0,255), thick=2):
    for x1,y1,x2,y2,_ in boxes:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)
    return frame

# -------------- main --------------

def main():
    args = parse_args()
    base = os.path.abspath(args.base)
    artifacts = os.path.join(base, "artifacts")
    onnx_path = os.path.join(artifacts, "model.onnx")
    if not os.path.exists(onnx_path):
        # many packs keep a copy under model/
        alt = os.path.join(base, "model", "model.onnx")
        if os.path.exists(alt):
            onnx_path = alt
        else:
            raise SystemExit(f"ONNX not found in {artifacts}/model.onnx or {base}/model/model.onnx")

    cfg = load_cfg(base)
    size = get_input_size(cfg, args.input_size)
    score_thr = get_score_thr(cfg, args.score)
    letterbox_on, pad_mode, pad_color, reverse = get_letterbox_mode(cfg, args.pad_center)
    if args.rgb:
        reverse = True

    # TIDL EP session
    provider_options = [{
        "artifacts_folder": artifacts,
        "platform": "J7",                           # TDA4VM family
        "tidl_tools_path": "/opt/edgeai/tidl_tools",
        "debug_level": "0",
    }]
    sess = ort.InferenceSession(
        onnx_path,
        providers=[("TIDLExecutionProvider", provider_options), "CPUExecutionProvider"]
    )
    in_name = sess.get_inputs()[0].name

    # input source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        title = f"TIDL Face – {os.path.basename(args.video)}"
    else:
        cap = cv2.VideoCapture(args.camera)
        title = f"TIDL Face – camera {args.camera}"

    if not cap.isOpened():
        raise SystemExit("ERROR: cannot open input source")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))
        if not writer.isOpened():
            print(f"WARNING: cannot open writer for {args.save}; disabling save")
            writer = None

    t0 = time.time(); frames = 0; fps_disp = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        inp, meta = preprocess_bgr(frame, size,
                                   reverse_channels=reverse,
                                   letterbox_on=letterbox_on,
                                   pad_mode=("center" if args.pad_center else pad_mode),
                                   pad_color=pad_color)
        outs = sess.run(None, {in_name: inp})
        boxes = postprocess(outs, meta, score_thr=score_thr, iou_thr=args.iou, letterbox_on=letterbox_on)

        frames += 1
        if frames >= 20:
            t1 = time.time(); fps_disp = frames / (t1 - t0); t0, frames = t1, 0
        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        frame = draw_boxes(frame, boxes, (0,0,255), 2)  # RED boxes

        if writer is not None:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow(title, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
