#!/usr/bin/env python3
"""
Live2D Service Module
=====================
Provides Live2D model rendering for the ESP32 AstrBot adapter.
Uses EGL for headless OpenGL rendering and outputs JPEG frames.

Architecture:
A background asyncio task (running on the same event loop as the WebSocket
handlers) continuously advances the Live2D model animation and writes the
latest frame into ``_last_frame``.  External callers obtain a frame via
``render_frame()`` which returns the cached JPEG instantly.

**No background OS threads are used** — all OpenGL/EGL operations stay on
the thread where ``glInit()`` was called, avoiding context-sharing issues.

Usage:

    from live2d_service import init_global_service, get_global_service, shutdown_global_service

    # 初始化（指定模型路径）
    init_global_service("/tmp/live2d_models/tomori/model.json")

    # 获取实例
    svc = get_global_service()

    # 获取最新帧（非阻塞）
    jpeg_bytes = svc.render_frame()

    # 解析 LLM 回复中的动作标签
    text, motion, expr = svc.parse_tags(llm_response)
    if motion:
        svc.start_motion(motion)

    svc.stop()
"""

import sys
import os
import re
import time
import threading
import asyncio
import subprocess
import json
import numpy as np
from PIL import Image
import io
from astrbot import logger
from OpenGL import EGL

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
    # __init__
    # ------------------------------------------------------------------
    def __init__(self, model_path=None, width=320, height=240,
                 fps=10, jpeg_quality=80):
        """
        Initialize the Live2D service.

        Args:
            model_path: Path to model.json or .model3.json file.
            width:      Render width  (default: 320 for ESP32 display).
            height:     Render height (default: 240).
            fps:        Target frame rate (default: 10).
            jpeg_quality: JPEG compression quality 1-100 (default: 80).
        """
        self.model_path = model_path
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
        self._render_task = None
        self._last_frame = None
        self._last_frame_time = 0.0
        self._lock = threading.Lock()
        self._event_loop = None

        # ---- EGL resources (created in start(), used in _render_and_cache) ----
        self._egl_display = None
        self._egl_surface = None
        self._egl_context = None

        self._current_motion = None
        self._current_expression = None
        self._last_emotion = None
        self._last_emotion_time = 0.0
        self._auto_idle_delay = 8.0  # 动作播放完后 N 秒自动回归 idle
        self._force_idle_next = False  # 标记下一帧强制回归 idle
        
        # ---- 可用动作 / 表情列表（从 model.json 解析） ----
        self._available_motions = []
        self._available_expressions = []
        
        # ---- 情绪分类系统 ----
        # EMOTION_CLASSES 在 _parse_available_from_model_json() 调用后自动生成
        self.EMOTION_CLASSES = {}       # {情绪分类: [动作名1, 动作名2, ...]}
        self.EXPRESSION_CLASSES = {}    # {情绪分类: [表情名1, 表情名2, ...]}
        self.EMOTION_TAG_PATTERN = re.compile(r'<emotion=([^>]+)>', re.IGNORECASE)
        self.EMOTION_SYNONYMS = {}      # 同义词映射表
        
        # ---- 实时推送回调（可选） ----
        self._push_handler = None

    # ------------------------------------------------------------------
    # 模型自动发现（静态方法）
    # ------------------------------------------------------------------
    @staticmethod
    def discover_models(models_dir: str) -> dict:
        """
        扫描模型目录，发现所有可用的 Live2D 模型。
        遍历 models_dir 下的每个子目录，查找 model.json 或 *.model3.json。

        Args:
            models_dir: 模型存放的根目录。

        Returns:
            dict: {模型名称(str): model.json完整路径(str)}
                  模型名称 = 子目录名。
        """
        models = {}
        if not models_dir or not os.path.isdir(models_dir):
            return models
        logger.info("[模型扫描] 扫描目录: %s", models_dir)
        for entry in sorted(os.listdir(models_dir)):
            entry_path = os.path.join(models_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            # 在子目录中查找模型定义文件
            for f in os.listdir(entry_path):
                if f == 'model.json' or f.endswith('.model3.json'):
                    models[entry] = os.path.join(entry_path, f)
                    logger.info("[模型扫描]  发现模型: %s → %s", entry, models[entry])
                    break
        logger.info("[模型扫描] 共发现 %d 个模型", len(models))
        return models

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def is_initialized(self) -> bool:
        """Return whether the service has been successfully started."""
        return self._initialized

    def set_push_handler(self, handler):
        """
        设置实时帧推送回调。

        Args:
            handler: 异步回调 ``async fn(jpeg_bytes: bytes)``
                     每渲染一帧就会被调用一次。设为 None 取消推送。
        """
        self._push_handler = handler
        if handler:
            logger.info("Live2D 实时推送回调已注册")
        else:
            logger.info("Live2D 实时推送回调已清除")

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
        """
        Initialize Live2D, EGL/OpenGL, load the model, and launch the
        async background render loop on the current event loop.
        """
        if self._initialized:
            logger.info("start() called but already initialized – no-op.")
            return

        if not self.model_path or not os.path.isfile(self.model_path):
            logger.error("模型文件不存在: %s", self.model_path)
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 根据模型文件后缀自动选择 v2 / v3 引擎
        if self.model_path.endswith('.model3.json'):
            import live2d.v3 as live2d
            logger.info("使用 Live2D Cubism v3 引擎")
        else:
            import live2d.v2 as live2d
            logger.info("使用 Live2D Cubism v2 引擎（兼容 .moc 格式）")
        self._live2d = live2d

        # Monkey-patch: 处理部分模型表情文件为空（0 bytes）导致 json.loads(b'') 报错
        _original_json_parse = live2d.platform_manager.PlatformManager.jsonParseFromBytes

        def _safe_json_parse(self_ptr, buf):
            if not buf or (isinstance(buf, bytes) and len(buf) == 0):
                return {}
            return _original_json_parse(self_ptr, buf)

        live2d.platform_manager.PlatformManager.jsonParseFromBytes = _safe_json_parse

        # Ensure a DISPLAY is set – required by Mesa's EGL/X11 platform
        if not os.environ.get('DISPLAY', ''):
            os.environ['DISPLAY'] = ':99'
            logger.info("DISPLAY not set; defaulted to ':99'")
        self._ensure_xvfb()

        # ── 手动创建 EGL OpenGL 上下文（live2d-py 不自建上下文） ──
        self._egl_display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        _major, _minor = EGL.EGLint(), EGL.EGLint()
        EGL.eglInitialize(self._egl_display, _major, _minor)
        _config_attrs = (EGL.EGLint * 13)(
            EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
            EGL.EGL_RED_SIZE, 8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE, 8,
            EGL.EGL_ALPHA_SIZE, 8,
            EGL.EGL_NONE)
        _cfg = (EGL.EGLConfig * 1)()
        _num = EGL.EGLint()
        EGL.eglChooseConfig(self._egl_display, _config_attrs, _cfg, 1, _num)
        _pb_attrs = (EGL.EGLint * 5)(
            EGL.EGL_WIDTH, self.width,
            EGL.EGL_HEIGHT, self.height,
            EGL.EGL_NONE)
        self._egl_surface = EGL.eglCreatePbufferSurface(self._egl_display, _cfg[0], _pb_attrs)
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        self._egl_context = EGL.eglCreateContext(self._egl_display, _cfg[0], EGL.EGL_NO_CONTEXT, None)
        EGL.eglMakeCurrent(self._egl_display, self._egl_surface, self._egl_surface, self._egl_context)
        import ctypes
        _egl_lib = ctypes.CDLL("libEGL.so.1")
        _egl_get_ctx = _egl_lib.eglGetCurrentContext
        _egl_get_ctx.restype = ctypes.c_void_p
        from OpenGL import platform
        platform.GetCurrentContext = _egl_get_ctx
        logger.info("EGL OpenGL 上下文创建成功 (Mesa %s)", _major.value)

        # ── 初始化 Live2D 框架 ──
        live2d.init()
        live2d.glInit()

        # ── 加载模型 ──
        self._model = live2d.LAppModel()
        # disable_precision=True: GLSL 1.20 不支持 `precision mediump float;`
        # （这是 OpenGL ES 语法，Mesa 的 GLSL 1.20 编译器不接受）
        self._model.LoadModelJson(self.model_path, disable_precision=True)

        # 解析 model.json 获取可用动作和表情
        self._parse_available_from_model_json()

        # 自动构建情绪分类系统（根据 motion/expression 名称关键词）
        self._build_emotion_classification()
        logger.info("[情绪] 自动分类完成: %d 个情绪类别",
                     len(self.EMOTION_CLASSES))

        # ---- viewport -------------------------------------------------
        _gl_viewport_set = False
        # Attempt 1 – glad (loaded inside live2d-py)
        try:
            self._live2d.glViewport(0, 0, self.width, self.height)
            _gl_viewport_set = True
            logger.info("glViewport(0, 0, %d, %d) set OK via glad", self.width, self.height)
        except AttributeError:
            logger.debug("glViewport not found via glad, trying ctypes fallback...")

        if not _gl_viewport_set:
            # Attempt 2 – ctypes fallback with correct arg types
            try:
                from ctypes import cdll, c_int
                libGL = cdll.LoadLibrary("libGL.so.1")
                glViewport_func = libGL.glViewport
                glViewport_func.restype = None
                glViewport_func.argtypes = [c_int, c_int, c_int, c_int]
                glViewport_func(0, 0, self.width, self.height)
                _gl_viewport_set = True
                logger.info("glViewport(0, 0, %d, %d) set OK via ctypes", self.width, self.height)
            except Exception as exc:
                logger.warning(
                    "Could not set glViewport via ctypes fallback: %s. "
                    "Rendering may produce incorrect dimensions.", exc
                )
        # ---------------------------------------------------------------
        # ── 告诉 Live2D 引擎正确的渲染尺寸 ──
        # 必须调用！否则 Live2DGLWrapper.width/height 保持 0，
        # clip 管理器恢复 viewport 时会设成 0x0，导致渲染到零空间
        self._model.Resize(self.width, self.height)
        logger.info("模型 Resize(%d, %d)", self.width, self.height)
        # ---------------------------------------------------------------
        self._initialized = True
        self._last_frame_time = time.time()

        # Start idle motion with NORMAL priority (more visible animation)
        idle_name = self.get_idle_motion_name()
        if idle_name:
            try:
                self._model.StartMotion(idle_name, 0, live2d.MotionPriority.NORMAL)
                logger.info("空闲动作 '%s' 已启动 (priority=NORMAL).", idle_name)
            except Exception as e:
                logger.info("启动空闲动作 '%s' 失败: %s", idle_name, e)
        else:
            logger.info("模型没有可用的空闲动作")

        # 设置默认表情（如果模型有 idle 或 default 表情文件）
        for candidate in ['idle', 'default', 'smile01', '']:
            if not candidate:
                break
            if candidate in self._available_expressions:
                try:
                    self._model.SetExpression(candidate)
                    self._current_expression = candidate
                    logger.info("默认表情 '%s' 已设置", candidate)
                except Exception:
                    pass
                break

        # Render the very first frame synchronously NOW
        self._render_and_cache()
        if self._last_frame:
            logger.info("首帧已缓存: %d bytes", len(self._last_frame))
        else:
            logger.warning("首帧渲染返回空！检查 GL 配置。")

        # Start background async task for continuous rendering
        self._running = True
        try:
            self._event_loop = asyncio.get_event_loop()
            self._render_task = self._event_loop.create_task(self._async_render_loop())
            logger.info("异步渲染循环任务已创建。")
        except RuntimeError as e:
            logger.warning("无法获取事件循环: %s。仅支持静态渲染。", e)

        logger.info(
            "Live2DService 初始化完成 – model=%s size=%dx%d @%dfps "
            "motions=%d expressions=%d",
            self.model_path, self.width, self.height, self.fps,
            len(self._available_motions), len(self._available_expressions),
        )

    def stop(self):
        """Clean up Live2D resources and stop async rendering."""
        self._running = False
        self._push_handler = None  # 清除推送回调

        if self._render_task:
            self._render_task.cancel()
            self._render_task = None

        if self._initialized and self._live2d:
            try:
                self._live2d.dispose()
            except Exception as exc:
                logger.warning("Error during live2d.dispose(): %s", exc)

        self._initialized = False
        self._model = None
        self._live2d = None
        logger.info("Live2DService 已停止。")

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
    # Emotion（情绪）自动控制系统
    # 1. 根据 motion/expression 名称自动分类
    # 2. 触发 emotion 标签时从对应类别随机选一个播放
    # 3. 播放完自动回归 idle
    # ------------------------------------------------------------------
    def _build_emotion_classification(self):
        """
        根据模型的实际 motion/expression 名称，通过关键词自动归类到情绪类别。
        同时建立中英文同义词映射。
        """
        # ---- 情绪分类关键词规则 ----
        # motion 名称包含关键词中任意一个即归入该类
        CLASSIFICATION_RULES = {
            'happy':     ['smile', 'kandou', 'sing', 'gacha', 'happy', 'joy', 'laugh'],
            'sad':       ['sad', 'cry', 'crying', 'sorrow'],
            'angry':     ['angry', 'rage', 'annoy', 'irritated'],
            'surprised': ['surprised', 'shock', 'kime', 'amaze', 'wonder'],
            'shy':       ['shame', 'shy', 'embarrass', 'blush'],
            'thinking':  ['thinking', 'think', 'ponder', 'nf', 'nnf'],
            'serious':   ['serious', 'determin'],
            'bye':       ['bye', 'wave', 'goodbye'],
            'idle':      ['idle', 'breath', 'wait', 'stand'],
        }

        # ---- 同义词映射（中英文 + 变体） ----
        self.EMOTION_SYNONYMS = {
            '开心': 'happy', '高兴': 'happy', '快乐': 'happy', '笑': 'happy',
            '难过': 'sad', '伤心': 'sad', '悲伤': 'sad', '哭': 'sad',
            '生气': 'angry', '愤怒': 'angry', '怒': 'angry',
            '惊讶': 'surprised', '震惊': 'surprised', '吃惊': 'surprised', '惊喜': 'surprised',
            '害羞': 'shy', '不好意思': 'shy',
            '思考': 'thinking', '想': 'thinking',
            '认真': 'serious', '严肃': 'serious',
            '再见': 'bye', '拜拜': 'bye', '挥手': 'bye',
            '发呆': 'idle', '待机': 'idle',
            '喜悦': 'happy', '欢乐': 'happy', '愉快': 'happy',
            '忧愁': 'sad', '忧郁': 'sad',
            '狂喜': 'happy', '激动': 'happy', '兴奋': 'happy',
            '恐惧': 'surprised', '惊吓': 'surprised',
            '疑惑': 'thinking', '困惑': 'thinking', '犹豫': 'thinking',
            '爱': 'happy', '喜欢': 'happy', '心动': 'happy',
        }

        # 构建 motions 分类
        motion_classes = {}
        used_motions = set()

        for cat_name, keywords in CLASSIFICATION_RULES.items():
            matched = []
            for m in self._available_motions:
                if m in used_motions:
                    continue
                if any(k in m.lower() for k in keywords):
                    matched.append(m)
            if matched:
                motion_classes[cat_name] = matched
                used_motions.update(matched)

        # 剩余未分类的 motion 放到 'other' 类别
        unclassified = [m for m in self._available_motions if m not in used_motions]
        if unclassified:
            motion_classes['other'] = unclassified

        self.EMOTION_CLASSES = motion_classes

        # 构建 expressions 分类（规则同上）
        expr_classes = {}
        used_exprs = set()
        for cat_name, keywords in CLASSIFICATION_RULES.items():
            matched = []
            for e in self._available_expressions:
                if e in used_exprs:
                    continue
                if any(k in e.lower() for k in keywords):
                    matched.append(e)
            if matched:
                expr_classes[cat_name] = matched
                used_exprs.update(matched)

        unclassified_exprs = [e for e in self._available_expressions if e not in used_exprs]
        if unclassified_exprs:
            expr_classes['other'] = unclassified_exprs

        self.EXPRESSION_CLASSES = expr_classes

        logger.info(
            "[情绪分类] motions: %d 类 %d 个; expressions: %d 类 %d 个",
            len(motion_classes), len(self._available_motions),
            len(expr_classes), len(self._available_expressions),
        )

    def _get_emotion_category(self, emotion_name: str) -> str:
        """
        将用户输入的 emotion 名称映射到标准情绪类别。
        支持中文/英文/同义词/部分匹配。
        返回标准类别名（如 'happy'），无法映射返回 None。
        """
        name = emotion_name.lower().strip()

        # 1. 直接匹配类别名
        if name in self.EMOTION_CLASSES:
            return name

        # 2. 查询同义词映射
        if name in self.EMOTION_SYNONYMS:
            return self.EMOTION_SYNONYMS[name]

        # 3. 部分匹配（包含关系）
        for cat_name in self.EMOTION_CLASSES:
            if name in cat_name or cat_name in name:
                return cat_name

        # 4. 输入本身就是可用 motion 名，归入 'other'
        if name in self._available_motions:
            return 'other'

        return None

    def set_emotion(self, emotion_name: str) -> bool:
        """
        通过情绪名称自动触发动画：
        1. 自动映射到标准情绪类别
        2. 从该类随机选一个 motion 播放
        3. 如果有对应表情也自动设置
        4. 播放完后自动回归 idle
        """
        if not self._initialized or not self._model:
            logger.warning("set_emotion called but service not initialised.")
            return False

        import random
        category = self._get_emotion_category(emotion_name)

        if not category:
            logger.warning("无法识别情绪 '%s'，可用类别: %s",
                         emotion_name, self.get_available_emotions())
            return False

        motion_pool = self.EMOTION_CLASSES.get(category, [])
        expr_pool = self.EXPRESSION_CLASSES.get(category, [])

        if not motion_pool:
            logger.warning("情绪 '%s' 分类 '%s' 下没有可用动作", emotion_name, category)
            return False

        # 随机选 motion 和 expression
        chosen_motion = random.choice(motion_pool)
        chosen_expr = random.choice(expr_pool) if expr_pool else None

        logger.info("[情绪] %s → %s → motion=%s expr=%s",
                    emotion_name, category, chosen_motion, chosen_expr or '(无)')

        # FORCE 优先级打断当前动作
        self.start_motion(chosen_motion, self._live2d.MotionPriority.FORCE)

        # 设置表情（表情文件可能是 0 bytes，忽略错误）
        if chosen_expr:
            try:
                self._model.SetExpression(chosen_expr)
                self._current_expression = chosen_expr
            except Exception:
                pass

        # 记录状态用于自动回归 idle
        self._last_emotion = category
        self._last_emotion_time = time.time()
        self._force_idle_next = False

        return True

    def get_available_emotions(self) -> list:
        """返回所有可用的情绪类别名称列表（过滤掉 'other'）。"""
        return [c for c in self.EMOTION_CLASSES if c != 'other'] + (['other'] if 'other' in self.EMOTION_CLASSES else [])
    # ------------------------------------------------------------------
    # 可用动作 / 表情查询
    # ------------------------------------------------------------------
    def get_available_motions(self):
        """
        返回从 model.json 解析到的所有可用动作组名称列表。
        """
        motions = list(self._available_motions)
        if self._initialized and self._model:
            try:
                api_motions = list(self._model.GetMotionGroupNames())
                for m in api_motions:
                    if m not in motions:
                        motions.append(m)
            except Exception:
                pass
        return motions

    def get_available_expressions(self):
        """
        返回从 model.json 解析到的所有可用表情名称列表。
        """
        expressions = list(self._available_expressions)
        if self._initialized and self._model:
            try:
                api_exprs = list(self._model.GetExpressionNames())
                for e in api_exprs:
                    if e not in expressions:
                        expressions.append(e)
            except Exception:
                pass
        return expressions

    def get_idle_motion_name(self):
        """
        返回最适合作为空闲/待机动作的动作组名称。
        """
        motions = self.get_available_motions()
        if not motions:
            return None
        for candidate in ['idle01', 'idle', 'Idle', 'Idle01', 'Idle_0',
                          'breath', 'Breath', 'wait', 'Wait']:
            if candidate in motions:
                return candidate
        return motions[0]

    # ------------------------------------------------------------------
    # model.json 解析
    # ------------------------------------------------------------------
    def _parse_available_from_model_json(self):
        """
        直接读取 model.json 文件，解析 motions 和 expressions 列表。
        兼容 Live2D v2（model.json）和 v3（.model3.json）格式。
        """
        motions = []
        expressions = []

        try:
            with open(self.model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("[模型] 读取 %s 失败: %s", self.model_path, e)
            self._available_motions = motions
            self._available_expressions = expressions
            return

        # ---------- 提取 motions ----------
        motions_data = None
        raw = data.get('motions')
        if isinstance(raw, dict):
            motions_data = raw
        if not motions_data:
            raw = data.get('Motions')
            if isinstance(raw, dict):
                motions_data = raw
        if not motions_data:
            file_refs = data.get('FileReferences') or {}
            raw = file_refs.get('Motions')
            if isinstance(raw, dict):
                motions_data = raw

        if motions_data:
            motions = list(motions_data.keys())
            logger.info("[模型] 从 motions 解析到 %d 个动作组: %s",
                        len(motions), motions)

        # ---------- 提取 expressions ----------
        expr_list = None
        raw = data.get('expressions')
        if isinstance(raw, list):
            expr_list = raw
        if not expr_list:
            raw = data.get('Expressions')
            if isinstance(raw, list):
                expr_list = raw
        if not expr_list:
            file_refs = data.get('FileReferences') or {}
            raw = file_refs.get('Expressions')
            if isinstance(raw, list):
                expr_list = raw

        if expr_list:
            for expr in expr_list:
                if not isinstance(expr, dict):
                    continue
                name = (expr.get('name') or expr.get('Name') or
                        expr.get('Id') or '')
                if not name:
                    fname = (expr.get('file') or expr.get('FileName') or '')
                    name = (fname.replace('.exp3.json', '')
                            .replace('.exp.json', '')
                            .replace('.json', ''))
                if name:
                    expressions.append(name)
            logger.info("[模型] 从 expressions 解析到 %d 个表情: %s",
                        len(expressions), expressions)

        self._available_motions = motions
        self._available_expressions = expressions
        logger.info("[模型] 从 %s 解析完成: %d 个动作组, %d 个表情",
                    os.path.basename(self.model_path),
                    len(motions), len(expressions))

    # ------------------------------------------------------------------
    # Frame rendering – public API (non-blocking)
    # ------------------------------------------------------------------
    def render_frame(self):
        """
        Return the latest cached frame immediately (non-blocking).

        Returns:
            JPEG ``bytes``, or ``None`` if no frame has been rendered yet.
        """
        if not self._initialized or not self._model:
            return None
        return self._last_frame

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------
    def _render_and_cache(self):
        """
        Render one frame and update internal cache.
        **Must be called from the same thread as ``start()`` / ``glInit()``.**
        """
        if not self._initialized or not self._model:
            return None
    
        with self._lock:
            # ── 每次渲染前重新激活 EGL 上下文 ──
            ok = EGL.eglMakeCurrent(self._egl_display, self._egl_surface,
                                     self._egl_surface, self._egl_context)
            if not ok:
                err_code = EGL.eglGetError()
                logger.error(
                    "eglMakeCurrent failed (0x%x) before render — dpy=%s surf=%s ctx=%s",
                    err_code, self._egl_display, self._egl_surface, self._egl_context,
                )
                return self._last_frame
    
            try:
                self._model.Update()
                self._live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)
                self._model.Draw()
                
                # ── 自动回归 idle：触发动作播放完一段时间后回到 idle ──
                if (self._last_emotion_time > 0
                    and time.time() - self._last_emotion_time > self._auto_idle_delay
                    and self._current_motion != self.get_idle_motion_name()):
                    idle_name = self.get_idle_motion_name()
                    if idle_name:
                        self._model.StartMotion(idle_name, 0, self._live2d.MotionPriority.NORMAL)
                        self._current_motion = idle_name
                        self._current_expression = 'idle'
                        self._last_emotion = 'idle'
                        self._last_emotion_time = 0.0
                        logger.debug("自动回归空闲动作: %s", idle_name)
 
    
                # ── 用 glReadPixels 替代 live2d.v2 不支持的 readPixels ──
                from OpenGL import GL
                import ctypes
                _buf = (ctypes.c_ubyte * (self.width * self.height * 4))()
                GL.glReadPixels(0, 0, self.width, self.height,
                                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, _buf)
                pixels = bytes(_buf)
    
                if not pixels:
                    logger.warning("readPixels returned empty.")
                    return self._last_frame
    
                img_array = (
                    np.frombuffer(pixels, dtype=np.uint8)
                    .reshape(self.height, self.width, 4)
                )
                img_array = np.flipud(img_array)
                img = Image.fromarray(img_array[:, :, :3], 'RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=self.jpeg_quality)
                jpeg_bytes = buf.getvalue()
    
                self._last_frame = jpeg_bytes
                self._last_frame_time = time.time()
                return jpeg_bytes
    
            except Exception as e:
                logger.error("Render error: %s", e, exc_info=True)
                return self._last_frame

    # ------------------------------------------------------------------
    # Tag parsing
    # ------------------------------------------------------------------
    def parse_tags(self, text):
        """
        Parse Live2D control tags from LLM response text.

        Args:
            text: Raw LLM response text (may contain tags).

        Returns:
            ``(cleaned_text, motion_name, expression_name, emotion_name)``
        """
        if not text:
            return text, None, None, None

        motion = None
        expression = None
        emotion = None

        motion_match = self.MOTION_TAG_PATTERN.search(text)
        if motion_match:
            motion = motion_match.group(1).strip()

        expr_match = self.EXPRESSION_TAG_PATTERN.search(text)
        if expr_match:
            expression = expr_match.group(1).strip()

        emotion_match = self.EMOTION_TAG_PATTERN.search(text)
        if emotion_match:
            emotion = emotion_match.group(1).strip()

        cleaned = self.ALL_TAGS_PATTERN.sub('', text).strip()
        return cleaned, motion, expression, emotion

    # ------------------------------------------------------------------
    # Async continuous rendering
    # ------------------------------------------------------------------
    async def _async_render_loop(self):
        """
        Async background task that renders frames continuously.
        Runs on the SAME event loop (same thread) as ``glInit()``.

        If a ``_push_handler`` is registered, each frame is pushed
        to it immediately.
        """
        logger.info("=== Async render loop started ===")
        while self._running:
            t_start = time.time()

            self._render_and_cache()

            if self._last_frame and self._push_handler:
                try:
                    await self._push_handler(self._last_frame)
                except Exception as e:
                    logger.info("Push handler error: %s", e)

            elapsed = time.time() - t_start
            sleep_time = max(0.0, self.frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        logger.info("Async render loop stopped.")


# ===================================================================
# Global singleton helpers
# ===================================================================
_global_service = None
_global_lock = threading.Lock()


def init_global_service(model_path: str) -> Live2DService:
    """
    使用指定的模型路径初始化全局 Live2DService 实例。
    如果已有实例运行，会先停止旧实例再创建新实例。

    Args:
        model_path: model.json 或 .model3.json 的完整路径。

    Returns:
        Live2DService 实例（已启动）。
    """
    global _global_service
    with _global_lock:
        if _global_service is not None:
            logger.info("停止旧 Live2D 服务以加载新模型...")
            _global_service.stop()
            _global_service = None

        logger.info("初始化 Live2D 服务: model_path=%s", model_path)
        _global_service = Live2DService(model_path=model_path)
        _global_service.start()
        return _global_service


def get_global_service() -> Live2DService:
    """
    获取全局 Live2DService 实例。
    必须先调用 init_global_service() 初始化，否则抛出 RuntimeError。

    Returns:
        Live2DService 实例。
    """
    global _global_service
    if _global_service is None:
        raise RuntimeError(
            "Live2DService 未初始化。请先调用 init_global_service(model_path) 初始化。"
        )
    return _global_service


def shutdown_global_service():
    """Shut down and clear the global Live2DService instance."""
    global _global_service
    with _global_lock:
        if _global_service is not None:
            _global_service.stop()
            _global_service = None
        logger.info("全局 Live2D 服务已关闭。")


# ===================================================================
# Standalone test
# ===================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Live2DService — 模型自动发现测试")
    print("=" * 60)

    # 测试模型发现
    test_dir = "/tmp/live2d_models"
    print(f"\n扫描目录: {test_dir}")
    found = Live2DService.discover_models(test_dir)
    if found:
        print(f"发现 {len(found)} 个模型:")
        for name, path in found.items():
            print(f"  [{name}] → {path}")
    else:
        print(f"  未发现模型，请将模型放入 {test_dir} 下的子目录中")

    print("\nDone.")
