
# import face_recognition
import cv2
import sys, os
import random


def bound_face_on_image(img, cascade=None):
    """
    Draws bounding boxes around faces in the given image.
    Returns the image with boxes drawn.
    """
    if img is None:
        raise ValueError("Input image is None")
    if cascade is None:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=6,
        minSize=(60, 60)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img


def extract_video(video_file, target_fps=None):
    frames = []
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return frames

    # Get original video's frame rate and total frame count
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If no target fps specified, use original fps
    if target_fps is None:
        target_fps = original_fps
    
    # Calculate the frame interval to achieve target fps
    frame_interval = max(1, int(original_fps / target_fps))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames to read or an error occurred
            break
            
        # Only keep frames at the calculated interval
        if frame_count % frame_interval == 0:
            frames.append(frame)
            
        frame_count += 1

    cap.release()
    print(f"Original FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Total original frames: {total_frames}")
    print(f"Extracted frames: {len(frames)}")
    return frames

def show_metrics(image, values):
    # metrics = ["Att 1", "Att 2", "Att 3", "Att 4", "Att 5"]

    panel_height = 80
    panel_color = (0, 0, 0)
    img_with_panel = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=panel_height,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=panel_color
    )

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    color = (255, 255, 255)
    thickness = 2
    y = img_with_panel.shape[0] - int(panel_height * 0.4)
    x = 30
    spacing = 250  # horizontal space between metrics

    for i, (attr, val) in enumerate(values.items()):
        text = f"{attr}: {val:.2f}"
        cv2.putText(img_with_panel, text, (x + i * spacing, y), font_face, scale, color, thickness, cv2.LINE_AA)

    return img_with_panel


def show_emotion(image):
    # Random placeholder
    emotions = ["Happy", "Sad", "Neutral", "Disgust", "Anger"]
    emotion = random.choice(emotions)
    
    # TO-DO: get the emotion from the model

    # Area for emotion text
    border_height = 100
    bottom_panel_height = 200
    border_color = (0, 0, 0)
    img_with_border = cv2.copyMakeBorder(
        image,
        top=border_height,
        bottom=bottom_panel_height,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color
    )

    # Place the emotion text on the border
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2      
    color = (255, 255, 255) 
    thickness = 4  
    pos = (50, int(border_height * 0.7))  

    # Draw white text
    cv2.putText(img_with_border, emotion, pos, font_face, scale, color, thickness, cv2.LINE_AA)
    return img_with_border





if __name__ == "__main__":
    video_files = [
        "edgeai-test-data/videos/video1_1280_768_h264.mp4",
        "VideoSet/video/1.mp4",
        "VideoSet/video/2.mp4",
        "VideoSet/video/3.mp4",
        "VideoSet/video/4.mp4",
        "VideoSet/video/5.mp4"
    ]

    video_file = video_files[3]
    
    print(video_file)

    extracted_frames = extract_video(video_file, target_fps=25)
    print("Extracted frames!")

    if extracted_frames:
        print(f"Successfully extracted {len(extracted_frames)} frames.")
        # cv2.imshow("First Frame", extracted_frames[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No frames were extracted.")


    # Create a window
    window_name = "Video Playback"
    cv2.namedWindow(window_name)
    print("Window playback")
    

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Cascade")

    print("Starting loop")
    for frame in extracted_frames:
        frame_with_boxes = bound_face_on_image(frame, cascade)
        frame_with_emotion = show_emotion(frame_with_boxes)
        # Example metrics; replace with real values as needed
        metrics = {"Confidence": random.uniform(0.5, 1.0)}
        frame_with_metrics = show_metrics(frame_with_emotion, metrics)
        cv2.imshow(window_name, frame_with_metrics)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key != 255:
            cv2.waitKey(0)
            cv2.waitKey(0)
    cv2.destroyAllWindows()



