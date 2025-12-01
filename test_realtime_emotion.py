#!/usr/bin/env python3


import sys
import cv2
from get_crop_points import YOLOXFaceDetector
from crop_face import FaceCropper
from predict_emotion import EmotionPredictor


def main():
    # CHANGE THIS PATH TO VIDEO FILE
    if len(sys.argv) < 2:
        print("Usage: python test_realtime_emotion.py path/to/video.mp4")
        sys.exit(1)

    # CHANGE THIS PATH TO VIDEO FILE
    video_path = sys.argv[1]
    
    # IF UI USING SMTH DIFFERENT FROM OPENCV, ADJUST HERE
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.2f} FPS")

    # Initialize models
    # OBJECTS HANDLING FACE DETECTION/CROPPING AND EMOTION PREDICTION
    print("Loading YOLOX face detector...")
    detector = YOLOXFaceDetector()
    
    print("Loading emotion predictor...")
    emotion_predictor = EmotionPredictor()
    
    cropper = FaceCropper(margin=0.15, out_size=(224, 224))

    print("\nProcessing frames...\n")
    
    frame_num = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Detect best face (returns empty list or single-element list)
        detections = detector.detect(frame, return_crops=False)
        
        if not detections:
            print(f"Frame {frame_num:04d}: No face detected")
            continue
        
        det = detections[0]
        detected_count += 1
        
        # Crop face using FaceCropper (returns PIL Image in RGB)
        face_pil = cropper.crop_face(frame, det)
        
        if face_pil is None:
            print(f"Frame {frame_num:04d}: Invalid crop area")
            continue
        
        # Predict emotion
        emotion = emotion_predictor.predict_emotion(face_pil)
        confidence = emotion_predictor.predict_confidence(face_pil)
        
        print(f"Frame {frame_num:04d}: {emotion} (confidence: {confidence:.3f})")
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Total frames: {frame_num}")
    print(f"  Faces detected: {detected_count}")
    print(f"  Detection rate: {100.0 * detected_count / frame_num:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
