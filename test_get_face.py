#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pathlib import Path
from get_crop_points import YOLOXFaceDetector

def main():
    parser = argparse.ArgumentParser(description="Run face detection and optionally save crops.")
    parser.add_argument('--save-crops', action='store_true', help='Enable saving crops to disk (requires --save-dir)')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save crops. Must be provided with --save-crops')
    args = parser.parse_args()

    # By default saving is disabled. To enable saving, both flags must be provided.
    if args.save_crops and not args.save_dir:
        parser.error("--save-crops requires --save-dir to be set")
    if args.save_dir and not args.save_crops:
        parser.error("--save-dir requires --save-crops to be set")

    save_crops = bool(args.save_crops and args.save_dir)
    save_dir_arg = Path(args.save_dir) if args.save_dir else None

    # Create window first
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', 1280, 720)
    # Create crops window (will display tiled thumbnails)
    cv2.namedWindow('Crops', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Crops', 800, 600)
    
    # Initialize video capture
    video_path = "VideoSet/video/1.mp4" # TO-DO: make this a CLI input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    writer = None

    # Initialize face detector (adjust base path to your YOLOX model folder)
    det = YOLOXFaceDetector(
        base="model_zoo/ONR-OD-8420-yolox-s-lite-mmdet-widerface-640x640",
        score_thr=0.25
    )

    frame_count = 0

    # Save all cropped face images
    crops_accum = []
    saved_count = 0

    if save_crops:
        save_dir = save_dir_arg
        save_dir = save_dir if save_dir is not None else Path("crops")
        save_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            # end of video
            break
        frame_count += 1

        # Run face detection and also retrieve crops for all detections this frame
        res = det.detect(frame, return_crops=True, return_all=True)
        if isinstance(res, tuple):
            boxes, crops = res
        else:
            boxes = res
            crops = []

        # save crops to accumulator (filter out None) and persist to disk (if enabled)
        for c in crops:
            if c is not None:
                crops_accum.append(c)
                if save_crops:
                    try:
                        fname = save_dir / f"crop_{frame_count:06d}_{saved_count:06d}.png"
                        cv2.imwrite(str(fname), c)
                        saved_count += 1
                    except Exception:
                        pass

        # DEBUGGING ONLY
        # Create and show a tiled contact-sheet of collected crops so far
        # def make_contact_sheet(imgs, thumb_size=(160, 160), cols=5, bg_color=(50, 50, 50)):
        #     if not imgs:
        #         h, w = thumb_size[1], thumb_size[0] * cols
        #         sheet = np.zeros((h, w, 3), dtype=np.uint8)
        #         sheet[:] = bg_color
        #         return sheet
        #     thumbs = []
        #     for im in imgs:
        #         try:
        #             t = cv2.resize(im, thumb_size)
        #         except Exception:
        #             # if resize fails, skip
        #             continue
        #         if t.ndim == 2:
        #             t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        #         thumbs.append(t)
        #     if not thumbs:
        #         h, w = thumb_size[1], thumb_size[0] * cols
        #         sheet = np.zeros((h, w, 3), dtype=np.uint8)
        #         sheet[:] = bg_color
        #         return sheet
        #     rows = int(np.ceil(len(thumbs) / cols))
        #     thumb_h, thumb_w = thumb_size[1], thumb_size[0]
        #     sheet_h = rows * thumb_h
        #     sheet_w = cols * thumb_w
        #     sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        #     sheet[:] = bg_color
        #     for idx, th in enumerate(thumbs):
        #         r = idx // cols
        #         c = idx % cols
        #         y1 = r * thumb_h
        #         x1 = c * thumb_w
        #         sheet[y1:y1+thumb_h, x1:x1+thumb_w] = th
        #     return sheet

        # sheet = make_contact_sheet(crops_accum, thumb_size=(160, 160), cols=5)
        # cv2.imshow('Crops', sheet)

        # Draw boxes on the frame
        # Only draw the highest-confidence box returned (detector now returns at most one)
        if boxes:
            box = boxes[0]
            x1, y1 = box['x1'], box['y1']
            x2, y2 = box['x2'], box['y2']
            # Draw rectangle (BGR format - using red)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Show frame
        cv2.imshow('Face Detection', frame)
        
        # Small delay between frames (1ms)
        cv2.waitKey(1)

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    # Close any OpenCV windows and exit immediately (non-blocking)
    cv2.destroyAllWindows()

    # Print final summary
    if len(crops_accum) > 0:
        if save_crops:
            print(f"Collected {len(crops_accum)} cropped faces (saved {saved_count} to {save_dir.resolve()}).")
        else:
            print(f"Collected {len(crops_accum)} cropped faces (saving disabled).")

if __name__ == "__main__":
    main()
