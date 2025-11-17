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

VIDEO_PATH = "VideoSet/video/1.mp4"
WINDOW_NAME = "Video Player Test"

BAR_HEIGHT = 30      # pixels for the timeline bar
BUTTON_HEIGHT = 40   # pixels for control buttons above the bar


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    player = param["player"]
    width = param["width"]
    height = param["height"]
    bar_height = param["bar_height"]
    buttons = param["buttons"]
    running_flag = param["running"]

    bar_top = height - bar_height

    # Scrub
    if y >= bar_top:
        x_ratio = x / float(width)
        t, idx = player.frame_at_bar_position(x_ratio)
        player.seek(t)
        print(f"[Mouse] Scrub: ratio={x_ratio:.2f}, time={t:.2f}, frame={idx}")
        return

    # Button triggers
    for btn in buttons:
        x1, y1, x2, y2 = btn["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            label = btn["label"]
            state = player.get_state()
            print(f"[Mouse] Button clicked: {label}")

            if label == "<-":
                player.rewind(0.2)

            elif label == "Play/Pause":
                if state["is_playing"]:
                    player.pause()
                else:
                    player.play()

            elif label == "Restart":
                player.restart()

            elif label == "->":
                player.fast_forward(0.2)

            elif label == "Quit":
                running_flag[0] = False

            break


def main():
    clip = load_video_to_clip(VIDEO_PATH)
    if clip.frame_count == 0:
        print("No frames loaded.")
        return

    player = VideoPlayer(clip)
    player.pause() 

    h, w = clip.frames[0].shape[:2]

    cv2.namedWindow(WINDOW_NAME)

    button_labels = ["<-", "Play/Pause", "Restart", "->", "Quit"]
    num_buttons = len(button_labels)
    button_width = w // num_buttons

    # --- Buttons layout ---
    buttons = []
    btn_y1 = h - BAR_HEIGHT - BUTTON_HEIGHT
    btn_y2 = h - BAR_HEIGHT - 1

    for i, label in enumerate(button_labels):
        x1 = i * button_width
        x2 = (i + 1) * button_width - 1
        buttons.append({
            "label": label,
            "rect": (x1, btn_y1, x2, btn_y2),
        })

    running = [True]

    params = {
        "player": player,
        "width": w,
        "height": h,
        "bar_height": BAR_HEIGHT,
        "buttons": buttons,
        "running": running,
    }
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, params)

    print("Loaded video:")
    print("  frames:", clip.frame_count)
    print("  fps:", clip.fps)
    print("  duration:", clip.duration)
    print()
    print("Use mouse:")
    print("  - Click buttons")
    print("  - Click bar to scrub")

    while running[0]:
        frame = player.current_frame()
        display = frame.copy()

        state = player.get_state()
        current_time = state["current_time"]
        duration = state["duration"]
        is_playing = state["is_playing"]
        current_idx = state["current_frame_index"]

        for btn in buttons:
            x1, y1, x2, y2 = btn["rect"]
            label = btn["label"]

            cv2.rectangle(display, (x1, y1), (x2, y2),
                          (70, 70, 70), thickness=-1)

            if label == "Play/Pause" and is_playing:
                cv2.rectangle(display, (x1, y1), (x2, y2),
                              (120, 120, 120), thickness=-1)

            cv2.rectangle(display, (x1, y1), (x2, y2),
                          (200, 200, 200), thickness=1)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tw, th = text_size
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(display, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        ratio = current_time / duration if duration > 0 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        bar_top = h - BAR_HEIGHT
        bar_bottom = h - 1
        progress_x = int(ratio * (w - 1))

        cv2.rectangle(display, (0, bar_top), (w - 1, bar_bottom),
                      (50,50,50), thickness=-1)

        cv2.rectangle(display, (0, bar_top), (progress_x, bar_bottom),
                      (150,150,150), thickness=-1)

        cv2.line(display, (progress_x, bar_top), (progress_x, bar_bottom),
                 (255,255,255), thickness=2)

        text = f"{current_time:.2f}s / {duration:.2f}s | frame {current_idx+1}/{clip.frame_count}"
        cv2.putText(display, text, (10, bar_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(WINDOW_NAME, display)

        cv2.waitKey(1)

        if is_playing:
            time.sleep(1.0 / clip.fps)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
