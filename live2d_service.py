#!/usr/bin/env python3
"""
Live2D Service Module
=====================
Provides Live2D model rendering for the ESP32 AstrBot adapter.
Uses EGL for headless OpenGL rendering and outputs JPEG frames.

Usage:
    from live2d_service import Live2DService
    service = Live2DService(model_path="/path/to/model.model3.json")
    service.start()
    
    # Render a frame
    jpeg_bytes = service.render_frame()
    
    # Send to ESP32 via WebSocket
    await websocket.send(jpeg_bytes, binary=True)
    
    # Parse LLM response for motion tags
    text, motion = service.parse_tags(llm_response)
    if motion:
        service.start_motion(motion)
    
    service.stop()
"""

import sys
import os
import re
import time
import threading
import numpy as np
from PIL import Image
import io

# Add live2d package path (adjust as needed)
LIVE2D_PACKAGE_PATH = "/tmp/live2d_py-0.6.1.1/package"
if LIVE2D_PACKAGE_PATH not in sys.path:
    sys.path.insert(0, LIVE2D_PACKAGE_PATH)


class Live2DService:
    """Manages Live2D model lifecycle and frame rendering for ESP32 display."""

    # Motion tag patterns for LLM output parsing
    MOTION_TAG_PATTERN = re.compile(r'<motion=([^>]+)>', re.IGNORECASE)
    EXPRESSION_TAG_PATTERN = re.compile(r'<expression=([^>]+)>', re.IGNORECASE)
    ALL_TAGS_PATTERN = re.compile(r'<[^>]+>')

    # Default model path (Hiyori sample from Cubism SDK)
    DEFAULT_MODEL_PATH = "/tmp/hiyori_model/Hiyori.model3.json"

    def __init__(self, model_path=None, width=320, height=240,
                 fps=10, jpeg_quality=80):
        """
        Initialize the Live2D service.

        Args:
            model_path: Path to .model3.json file
            width: Render width (default: 320 for ESP32 display)
            height: Render height (default: 240)
            fps: Target frame rate (default: 10)
            jpeg_quality: JPEG compression quality 1-100 (default: 80)
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.jpeg_quality = jpeg_quality

        # Internal state
        self._live2d = None
        self._model = None
        self._initialized = False
        self._running = False
        self._render_thread = None
        self._last_frame = None
        self._last_frame_time = 0
        self._lock = threading.Lock()

        # Motion state
        self._current_motion = None
        self._current_expression = None

    def start(self):
        """Initialize Live2D, EGL/OpenGL, and load the model."""
        if self._initialized:
            return

        import live2d.v3 as live2d

        self._live2d = live2d

        # Set DISPLAY if needed for EGL
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':99'

        # Initialize Live2D framework
        live2d.init()
        live2d.glInit()

        # Load model
        self._model = live2d.LAppModel()
        self._model.LoadModelJson(self.model_path)

        # Set viewport via OpenGL
        from ctypes import cdll
        libGL = cdll.LoadLibrary("libGL.so.1")
        glViewport = libGL.glViewport
        glViewport.restype = None
        glViewport.argtypes = [int, int, int, int]
        glViewport(0, 0, self.width, self.height)

        self._initialized = True
        self._last_frame_time = time.time()

        # Start idle motion
        try:
            self._model.StartMotion("Idle", 0, live2d.MotionPriority.IDLE)
        except Exception:
            pass  # Some models may not have Idle motions

        print(f"[Live2DService] Initialized: {self.model_path}")
        print(f"[Live2DService] Render size: {self.width}x{self.height} @ {self.fps}fps")

    def stop(self):
        """Clean up Live2D resources."""
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=2)
            self._render_thread = None
        if self._initialized and self._live2d:
            try:
                self._live2d.dispose()
            except Exception:
                pass
        self._initialized = False
        print("[Live2DService] Stopped.")

    def start_motion(self, motion_name, priority=None):
        """
        Start a motion on the Live2D model.

        Args:
            motion_name: Name of the motion (e.g., 'TapBody', 'Idle')
            priority: Motion priority (default: FORCE for external triggers)
        """
        if not self._initialized or not self._model:
            return

        if priority is None:
            priority = self._live2d.MotionPriority.FORCE

        try:
            self._model.StartMotion(motion_name, 0, priority)
            self._current_motion = motion_name
            print(f"[Live2DService] Started motion: {motion_name}")
        except Exception as e:
            print(f"[Live2DService] Failed to start motion '{motion_name}': {e}")

    def set_expression(self, expression_name):
        """Set facial expression on the Live2D model."""
        if not self._initialized or not self._model:
            return
        try:
            self._model.SetExpression(expression_name)
            self._current_expression = expression_name
            print(f"[Live2DService] Set expression: {expression_name}")
        except Exception as e:
            print(f"[Live2DService] Failed to set expression '{expression_name}': {e}")

    def render_frame(self):
        """
        Render a single frame and return JPEG bytes.

        Returns:
            bytes: JPEG image data, or None if rendering failed.
        """
        if not self._initialized or not self._model:
            return None

        with self._lock:
            try:
                # Update model animation
                self._model.Update()

                # Clear buffer with transparent background
                self._live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)

                # Draw the model
                self._model.Draw()

                # Read pixels from OpenGL framebuffer
                pixels = self._live2d.readPixels(self.width, self.height)
                if not pixels:
                    return None

                # Convert to numpy array and flip vertically
                img_array = np.frombuffer(pixels, dtype=np.uint8) \
                    .reshape(self.height, self.width, 4)
                img_array = np.flipud(img_array)

                # Convert RGBA to RGB
                img = Image.fromarray(img_array[:, :, :3], 'RGB')

                # Save to JPEG in memory
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=self.jpeg_quality)
                jpeg_bytes = buf.getvalue()

                self._last_frame = jpeg_bytes
                self._last_frame_time = time.time()

                return jpeg_bytes

            except Exception as e:
                print(f"[Live2DService] Render error: {e}")
                return None

    def get_last_frame(self):
        """Get the last rendered frame (cached)."""
        return self._last_frame

    # --- Tag Parsing ---

    def parse_tags(self, text):
        """
        Parse Live2D control tags from LLM response text.

        Extracts motion and expression tags, then returns the cleaned text
        and any detected actions.

        Args:
            text: Raw LLM response text (may contain tags like <motion=wave>)

        Returns:
            tuple: (cleaned_text, motion_name, expression_name)
                   - cleaned_text: Text with all tags removed
                   - motion_name: Detected motion or None
                   - expression_name: Detected expression or None
        """
        if not text:
            return text, None, None

        motion = None
        expression = None

        # Extract motion tag
        motion_match = self.MOTION_TAG_PATTERN.search(text)
        if motion_match:
            motion = motion_match.group(1).strip()

        # Extract expression tag
        expr_match = self.EXPRESSION_TAG_PATTERN.search(text)
        if expr_match:
            expression = expr_match.group(1).strip()

        # Remove all tags from text
        cleaned_text = self.ALL_TAGS_PATTERN.sub('', text).strip()

        return cleaned_text, motion, expression

    # --- Continuous Rendering (for real-time streaming) ---

    def start_continuous_rendering(self, callback=None):
        """
        Start a background thread that continuously renders frames.

        Args:
            callback: Optional function to call with each rendered JPEG bytes.
                     If None, frames are stored in _last_frame only.
        """
        if self._render_thread and self._render_thread.is_alive():
            return

        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop,
            args=(callback,),
            daemon=True
        )
        self._render_thread.start()
        print("[Live2DService] Continuous rendering started.")

    def stop_continuous_rendering(self):
        """Stop the background rendering thread."""
        self._running = False

    def _render_loop(self, callback):
        """Background render loop."""
        while self._running:
            frame_start = time.time()
            jpeg_bytes = self.render_frame()

            if jpeg_bytes and callback:
                try:
                    callback(jpeg_bytes)
                except Exception as e:
                    print(f"[Live2DService] Callback error: {e}")

            # Maintain target framerate
            elapsed = time.time() - frame_start
            sleep_time = max(0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # --- Utility ---

    def get_available_motions(self):
        """Get list of available motion group names."""
        if not self._initialized or not self._model:
            return []
        try:
            return list(self._model.GetMotionGroupNames())
        except Exception:
            return []


# --- Standalone test ---
if __name__ == "__main__":
    print("Live2DService - Standalone Test")
    print("=" * 40)

    service = Live2DService(fps=5, jpeg_quality=85)
    service.start()

    print("Available motions:", service.get_available_motions())

    # Test tag parsing
    test_texts = [
        "Hello! <motion=wave> How are you?",
        "<motion=TapBody><expression=happy> Great!",
        "No tags here",
        "<motion=bye> Goodbye! <expression=sad>"
    ]

    print("\nTag Parsing Tests:")
    for t in test_texts:
        clean, motion, expr = service.parse_tags(t)
        print(f"  Input:  {t}")
        print(f"  Output: '{clean}' | motion={motion} | expr={expr}")
        print()

    # Render a few test frames
    print("Rendering test frames...")
    for i in range(3):
        jpeg = service.render_frame()
        if jpeg:
            with open(f"/tmp/live2d_frames/service_test_{i}.jpg", "wb") as f:
                f.write(jpeg)
            print(f"  Frame {i}: {len(jpeg)} bytes saved.")

    service.stop()
    print("\nDone!")
