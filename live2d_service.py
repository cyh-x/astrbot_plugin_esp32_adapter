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
import subprocess
import numpy as np
from PIL import Image
import io

from astrbot import logger

# ---------------------------------------------------------------------------
# Live2D package discovery
# ---------------------------------------------------------------------------
LIVE2D_PACKAGE_PATH = os.environ.get(
    "LIVE2D_PACKAGE_PATH",
    "/tmp/live2d_py-0.6.1.1/package"
)
if LIVE2D_PACKAGE_PATH not in sys.path:
    sys.path.insert(0, LIVE2D_PACKAGE_PATH)


# ---------------------------------------------------------------------------
# Live2DService
# ---------------------------------------------------------------------------
class Live2DService:
    """Manages Live2D model lifecycle and frame rendering for ESP32 display."""

    # Motion / expression tag patterns for LLM output parsing
    MOTION_TAG_PATTERN = re.compile(r'<motion=([^>]+)>', re.IGNORECASE)
    EXPRESSION_TAG_PATTERN = re.compile(r'<expression=([^>]+)>', re.IGNORECASE)
    ALL_TAGS_PATTERN = re.compile(r'<[^>]+>')

    # ------------------------------------------------------------------
    # Default paths
    # ------------------------------------------------------------------
    DEFAULT_MODEL_PATH = os.environ.get(
        "LIVE2D_MODEL_PATH",
        "/tmp/hiyori_model/Hiyori.model3.json"
    )

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    def __init__(self, model_path=None, width=320, height=240,
                 fps=10, jpeg_quality=80):
        """
        Initialize the Live2D service.

        Args:
            model_path: Path to .model3.json file.
            width:      Render width  (default: 320 for ESP32 display).
            height:     Render height (default: 240).
            fps:        Target frame rate (default: 10).
            jpeg_quality: JPEG compression quality 1-100 (default: 80).
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.jpeg_quality = jpeg_quality

        # ---- internal state ----
        self._live2d = None
        self._model = None
        self._initialized = False
        self._running = False
        self._render_thread = None
        self._last_frame = None
        self._last_frame_time = 0.0
        self._lock = threading.Lock()

        # ---- motion / expression state ----
        self._current_motion = None
        self._current_expression = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def is_initialized(self) -> bool:
        """Return whether the service has been successfully started."""
        return self._initialized

    # ------------------------------------------------------------------
    # Xvfb management
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_xvfb():
        """Start Xvfb if it is not already running on the current DISPLAY."""
        display = os.environ.get('DISPLAY', ':99').strip()
        if not display:
            display = ':99'
        display_num = display.lstrip(':')

        # Check whether Xvfb is already listening on the display socket
        try:
            import socket
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.5)
            result = s.connect_ex(f'/tmp/.X11-unix/X{display_num}')
            s.close()
            if result == 0:
                logger.debug("Xvfb already running on display %s", display)
                return
        except Exception:
            pass

        logger.info("Xvfb not detected on display %s – starting it now ...", display)
        try:
            proc = subprocess.Popen(
                ['Xvfb', display, '-screen', '0', '1024x768x24'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give Xvfb a moment to boot
            import time as _time
            _time.sleep(0.5)

            # Quick sanity check
            ret = proc.poll()
            if ret is not None:
                logger.error("Xvfb exited immediately with code %d – check logs", ret)
            else:
                logger.info("Xvfb started (PID %d) on display %s", proc.pid, display)
        except FileNotFoundError:
            logger.warning("Xvfb binary not found – install with: apt-get install xvfb")
        except Exception as exc:
            logger.warning("Failed to start Xvfb: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Initialize Live2D, EGL/OpenGL and load the model."""
        if self._initialized:
            logger.info("start() called but already initialized – no-op.")
            return

        import live2d.v3 as live2d
        self._live2d = live2d

        # Ensure a DISPLAY is set – required by Mesa's EGL/X11 platform
        if not os.environ.get('DISPLAY', ''):
            os.environ['DISPLAY'] = ':99'
            logger.info("DISPLAY not set; defaulted to ':99'")
        self._ensure_xvfb()
        # Initialise Live2D framework + EGL-backed GL
        live2d.init()
        live2d.glInit()

        # Load model
        self._model = live2d.LAppModel()
        self._model.LoadModelJson(self.model_path)

        # ---- viewport -------------------------------------------------
        # Prefer glad-loaded glViewport (via EGL dispatch); fall back to
        # ctypes in case the glad symbol hasn't been exposed.
        _gl_viewport_set = False

        # Attempt 1 – glad (loaded inside live2d-py)
        try:
            # glad's glViewport may be accessible via live2d module if
            # the C extension exposes it.  Try the direct path first.
            self._live2d.glViewport(0, 0, self.width, self.height)
            _gl_viewport_set = True
        except AttributeError:
            pass

        if not _gl_viewport_set:
            # Attempt 2 – ctypes fallback
            try:
                from ctypes import cdll
                libGL = cdll.LoadLibrary("libGL.so.1")
                glViewport = libGL.glViewport
                glViewport.restype = None
                glViewport.argtypes = [int, int, int, int]
                glViewport(0, 0, self.width, self.height)
                _gl_viewport_set = True
            except Exception as exc:
                logger.warning(
                    "Could not set glViewport via ctypes fallback: %s. "
                    "Rendering may produce incorrect dimensions.", exc
                )

        if _gl_viewport_set:
            logger.info("glViewport(0, 0, %d, %d) set OK", self.width, self.height)

        # ---------------------------------------------------------------
        self._initialized = True
        self._last_frame_time = time.time()

        # Start idle motion (best-effort)
        try:
            self._model.StartMotion("Idle", 0, live2d.MotionPriority.IDLE)
            logger.debug("Idle motion started.")
        except Exception:
            pass  # model may not have Idle motions

        logger.info(
            "Live2DService initialised – model=%s size=%dx%d @%dfps",
            self.model_path, self.width, self.height, self.fps,
        )


    def stop(self):
        """Clean up Live2D resources."""
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=2)
            self._render_thread = None

        if self._initialized and self._live2d:
            try:
                self._live2d.dispose()
            except Exception as exc:
                logger.warning("Error during live2d.dispose(): %s", exc)

        self._initialized = False
        self._model = None
        self._live2d = None
        logger.info("Live2DService stopped.")

    # ------------------------------------------------------------------
    # Motion / expression control
    # ------------------------------------------------------------------
    def start_motion(self, motion_name, priority=None):
        """
        Start a motion on the Live2D model.

        Args:
            motion_name: Name of the motion group (e.g. 'TapBody', 'Idle').
            priority:    Motion priority constant from live2d.MotionPriority.
                         Defaults to ``FORCE`` for external triggers.
        """
        if not self._initialized or not self._model:
            logger.warning("start_motion called but service not initialised.")
            return

        if priority is None:
            priority = self._live2d.MotionPriority.FORCE

        try:
            self._model.StartMotion(motion_name, 0, priority)
            self._current_motion = motion_name
            logger.info("Started motion: %s (priority=%d)", motion_name, priority)
        except Exception as e:
            logger.error("Failed to start motion '%s': %s", motion_name, e)

    def set_expression(self, expression_name):
        """Set facial expression on the Live2D model."""
        if not self._initialized or not self._model:
            logger.warning("set_expression called but service not initialised.")
            return

        try:
            self._model.SetExpression(expression_name)
            self._current_expression = expression_name
            logger.info("Set expression: %s", expression_name)
        except Exception as e:
            logger.error("Failed to set expression '%s': %s", expression_name, e)

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------
    def render_frame(self):
        """
        Render a single frame and return JPEG bytes.

        Returns:
            JPEG ``bytes``, or ``None`` on failure.
        """
        if not self._initialized or not self._model:
            return None

        with self._lock:
            try:
                # Advance model animation
                self._model.Update()

                # Clear buffer (transparent background)
                self._live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)

                # Draw
                self._model.Draw()

                # Read GL framebuffer → raw RGBA bytes
                pixels = self._live2d.readPixels(self.width, self.height)
                if not pixels:
                    logger.warning("readPixels returned empty.")
                    return None

                # Convert raw bytes → numpy array (RGBA), flip Y
                img_array = (
                    np.frombuffer(pixels, dtype=np.uint8)
                    .reshape(self.height, self.width, 4)
                )
                img_array = np.flipud(img_array)

                # RGBA → RGB → JPEG in memory
                img = Image.fromarray(img_array[:, :, :3], 'RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=self.jpeg_quality)
                jpeg_bytes = buf.getvalue()

                self._last_frame = jpeg_bytes
                self._last_frame_time = time.time()
                return jpeg_bytes

            except Exception as e:
                logger.error("Render error: %s", e, exc_info=True)
                return None

    def get_last_frame(self):
        """Return the most recently rendered frame (cached)."""
        return self._last_frame

    # ------------------------------------------------------------------
    # Tag parsing
    # ------------------------------------------------------------------
    def parse_tags(self, text):
        """
        Parse Live2D control tags from LLM response text.

        Extracts ``<motion=...>`` and ``<expression=...>`` tags, then returns
        the cleaned text together with any detected action names.

        Args:
            text: Raw LLM response text (may contain tags).

        Returns:
            ``(cleaned_text, motion_name, expression_name)``
        """
        if not text:
            return text, None, None

        motion = None
        expression = None

        motion_match = self.MOTION_TAG_PATTERN.search(text)
        if motion_match:
            motion = motion_match.group(1).strip()

        expr_match = self.EXPRESSION_TAG_PATTERN.search(text)
        if expr_match:
            expression = expr_match.group(1).strip()

        cleaned = self.ALL_TAGS_PATTERN.sub('', text).strip()
        return cleaned, motion, expression

    # ------------------------------------------------------------------
    # Continuous rendering (background thread)
    # ------------------------------------------------------------------
    def start_continuous_rendering(self, callback=None):
        """
        Start a background thread that renders frames continuously.

        Args:
            callback: Optional callable ``fn(jpeg_bytes)`` invoked
                      with each new frame.  If not provided frames are
                      only stored in ``_last_frame``.
        """
        if self._render_thread and self._render_thread.is_alive():
            logger.info("Continuous rendering already running.")
            return

        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop,
            args=(callback,),
            daemon=True,
        )
        self._render_thread.start()
        logger.info("Continuous rendering started.")

    def stop_continuous_rendering(self):
        """Stop the background rendering thread."""
        self._running = False

    def _render_loop(self, callback):
        """Background render loop."""
        while self._running:
            t_start = time.time()
            jpeg_bytes = self.render_frame()
            if jpeg_bytes and callback:
                try:
                    callback(jpeg_bytes)
                except Exception as e:
                    logger.error("Callback error: %s", e)

            elapsed = time.time() - t_start
            sleep_time = max(0.0, self.frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_available_motions(self):
        """Return list of available motion group names."""
        if not self._initialized or not self._model:
            return []
        try:
            return list(self._model.GetMotionGroupNames())
        except Exception:
            return []


# ===================================================================
# Global singleton helpers
# ===================================================================
_global_service = None
_global_lock = threading.Lock()


def get_global_service() -> Live2DService:
    """Return the global Live2DService instance (lazy-initialised)."""
    global _global_service
    if _global_service is None:
        with _global_lock:
            if _global_service is None:
                _global_service = Live2DService()
                _global_service.start()
    return _global_service


def shutdown_global_service():
    """Shut down and clear the global Live2DService instance."""
    global _global_service
    with _global_lock:
        if _global_service is not None:
            _global_service.stop()
            _global_service = None


# ===================================================================
# Standalone test
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Live2DService — Standalone Test")
    print("=" * 50)

    svc = Live2DService(fps=5, jpeg_quality=85)
    svc.start()

    print("Available motions:", svc.get_available_motions())

    # ---- tag parsing demo ----
    test_texts = [
        "Hello! <motion=wave> How are you?",
        "<motion=TapBody><expression=happy> Great!",
        "No tags here",
        "<motion=bye> Goodbye! <expression=sad>",
    ]
    print("\nTag parsing tests:\n")
    for t in test_texts:
        clean, motion, expr = svc.parse_tags(t)
        print(f"  Input : {t}")
        print(f"  Output: '{clean}' | motion={motion} | expr={expr}\n")

    # ---- render test frames ----
    print("Rendering test frames ...")
    for i in range(3):
        jpeg = svc.render_frame()
        if jpeg:
            path = f"/tmp/live2d_frames/service_test_{i}.jpg"
            with open(path, "wb") as f:
                f.write(jpeg)
            print(f"  Frame {i}: {len(jpeg)} bytes → {path}")
        else:
            print(f"  Frame {i}: FAILED")

    svc.stop()
    print("\nDone.")
