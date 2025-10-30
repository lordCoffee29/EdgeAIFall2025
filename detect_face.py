# import face_recognition
import cv2
import sys, os

def bound_face(image_file):
    img = cv2.imread(image_file)

    if img is None:
        raise FileNotFoundError("Could not read file")
    
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=6,     # require more neighbor votes
        minSize=(60, 60)    # ignore tiny regions
    )

    # Draw red rectangles (BGR: red = (0,0,255))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)



    # # Load the image
    # image_path = image_file # Path to image
    # image = cv2.imread(image_path)

    # # Find all the faces in the image
    # face_locations = face_recognition.face_locations(image)

    # # Loop through each face found in the image
    # for (top, right, bottom, left) in face_locations:
    #     # Draw a red box around the face
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # BGR color for red

    # # Display the image with highlighted faces
    # cv2.imshow("Faces Highlighted", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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

if __name__ == "__main__":
    video_file = "edgeai-test-data/videos/video1_1280_768_h264.mp4"  # Replace with the actual path to your video file
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
    for i in range(len(extracted_frames)):
        frame = extracted_frames[i]
        
        # detect faces in grayscale copy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.20, 
            minNeighbors=6,     # require more neighbor votes
            minSize=(60, 60)    # ignore tiny regions
        )
        # draw red rectangles on the original frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # show the frame with boxes
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key != 255:
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()



