import time
import cv2
from video_player import VideoClip, VideoPlayer

def load_video_to_clip(path: str, fps: float = None) -> VideoClip:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not load video: {path}")
    
    # Use provided FPS if given, otherwise read from the file.
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    final_fps = fps if fps is not None else (file_fps if file_fps > 0 else 25.0)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return VideoClip(frames=frames, fps=final_fps)


# Load the clip
clip = load_video_to_clip("VideoSet/video/1.mp4")
print("Loaded video:")
print("  frames:", clip.frame_count)
print("  fps:", clip.fps)
print("  duration:", clip.duration)

# Create player
player = VideoPlayer(clip)

# Test playing
print("\n=== PLAY for 1.5 seconds ===")
player.play()
time.sleep(1.5)
print(player.get_state())

# Test pause
print("\n=== PAUSE ===")
player.pause()
print(player.get_state())

# Test fast-forward
print("\n=== FAST FORWARD 2 seconds ===")
print(player.fast_forward(2))

# Test rewind
print("\n=== REWIND 1 second ===")
print(player.rewind(1))

# Test seeking directly
print("\n=== SEEK to 0.5 seconds ===")
print(player.seek(0.5))

# Test current frame
print("\n=== CURRENT FRAME INDEX ===")
print("Frame Index:", player.current_frame_index())

