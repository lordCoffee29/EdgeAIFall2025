import time
from dataclasses import dataclass
from typing import List, Any

@dataclass
class VideoClip:
    frames: List[Any]
    fps: float = 25.0

    @property
    def frame_count(self):
        return len(self.frames)

    @property
    def duration(self):
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def frame_index_for_time(self, t: float):
        if self.frame_count == 0:
            return 0
        t = max(0.0, min(t, self.duration))
        idx = int(round(t * self.fps))
        return min(idx, self.frame_count - 1)

    def time_for_frame_index(self, idx: int):
        idx = max(0, min(idx, self.frame_count - 1))
        return idx / self.fps

    def get_frame(self, idx: int):
        idx = max(0, min(idx, self.frame_count - 1))
        return self.frames[idx]


class VideoPlayer:
    def __init__(self, clip: VideoClip):
        self.clip = clip

        # Start of video
        self._current_time = 0.0          
        self._is_playing = False
        self._last_play_wallclock = None

    def _sync_time(self):
        if not self._is_playing or self._last_play_wallclock is None:
            return

        now = time.time()
        elapsed = now - self._last_play_wallclock
        new_time = self._current_time + elapsed

        if new_time >= self.clip.duration:
            new_time = self.clip.duration
            self._is_playing = False
            self._last_play_wallclock = None

        self._current_time = new_time
        self._last_play_wallclock = now

    def _within_time(self):
        self._current_time = max(0.0, min(self._current_time, self.clip.duration))

    # Functions for controlling playback
    def play(self) -> dict:
        """Start/resume playback from current_time."""
        self._sync_time()
        if not self._is_playing and self._current_time < self.clip.duration:
            self._is_playing = True
            self._last_play_wallclock = time.time()
        return self.get_state()
    
    def pause(self) -> dict:
        """Pause playback."""
        self._sync_time()
        self._is_playing = False
        self._last_play_wallclock = None
        return self.get_state()
    
    def seek(self, target_time: float) -> dict:
        """Jump to a specific time in seconds."""
        self._sync_time()
        self._current_time = float(target_time)
        self._within_time()
        if self._is_playing:
            self._last_play_wallclock = time.time()
        return self.get_state()
    
    def fast_forward(self, delta: float = 5.0) -> dict:
        """Advance by delta seconds."""
        self._sync_time()
        self._current_time += delta
        self._within_time()
        if self._is_playing and self._current_time < self.clip.duration:
            self._last_play_wallclock = time.time()
        return self.get_state()
    
    def rewind(self, delta: float = 5.0) -> dict:
        """Go backwards by delta seconds."""
        self._sync_time()
        self._current_time -= delta
        self._within_time()
        if self._is_playing:
            self._last_play_wallclock = time.time()
        return self.get_state()
    
    def restart(self) -> dict:
        """Restart video from t = 0."""
        self._sync_time()
        self._current_time = 0.0
        if self._is_playing:
            self._last_play_wallclock = time.time()
        return self.get_state()
    
    def frame_at_bar_position(self, x_ratio: float):
        x_ratio = max(0.0, min(1.0, x_ratio))
        t = x_ratio * self.clip.duration
        idx = self.clip.frame_index_for_time(t)
        return t, idx
    

    # Functions to process video for emotion detection
    def current_timestamp(self) -> float:
        self._sync_time()
        return self._current_time
    
    def current_frame_index(self) -> int:
        self._sync_time()
        return self.clip.frame_index_for_time(self._current_time)
    
    def current_frame(self):
        idx = self.current_frame_index()
        return self.clip.get_frame(idx)

    

    # Justin: use/modify this to get video state/data into frontend
    def get_state(self) -> dict:
        self._sync_time()
        return {
            "current_time": self._current_time,
            "duration": self.clip.duration,
            "is_playing": self._is_playing,
            "current_frame_index": self.clip.frame_index_for_time(self._current_time)
        }