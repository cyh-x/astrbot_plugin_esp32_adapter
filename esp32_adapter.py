import asyncio
import json
import struct
import os
import time
import random
from typing import Dict, Optional

from .live2d_service import (
    init_global_service, get_global_service,
    shutdown_global_service, Live2DService
)

import websockets
from websockets import WebSocketServerProtocol
from astrbot.api.platform import (
    Platform, AstrBotMessage, MessageMember, PlatformMetadata, MessageType
)
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain, Image, Record
from astrbot.core.platform.astr_message_event import MessageSesion
from astrbot.api.platform import register_platform_adapter
from astrbot import logger

from .esp32_event import ESP32Event


# 二进制帧类型常量
FRAME_TYPE_OPUS_AUDIO = 0x00       # 上行：OPUS 音频数据
FRAME_TYPE_TTS_AUDIO = 0x01        # 下行：TTS 音频数据
FRAME_TYPE_IMAGE_CHUNK = 0x02      # 下行：图片分片数据
FRAME_TYPE_LIVE2D_FRAME = 0x03     # 下行：Live2D 帧


# 每个连接的会话状态
class DeviceSession:
    def __init__(self, websocket: WebSocketServerProtocol, device_id: str):
        self.websocket = websocket
        self.device_id = device_id
        self.audio_buffer = bytearray()
        self.audio_params = {}
        self.is_receiving_audio = False
        self.last_audio_time = 0
        self.audio_start_time = 0
        self.audio_timeout = 5.0  # 秒


@register_platform_adapter(
    "esp32",
    "ESP32 智能硬件适配器",
    default_config_tmpl={
        "host": "0.0.0.0",
        "ws_port": 8765,
        "auth_token": "",
        "audio_save_dir": "./esp32_audio",
        "max_audio_duration": 60,
        "live2d_injection_enabled": True,
        "live2d_injection_prompt": "【L2D_INJECTED】...",
        "live2d_model_name": "",
    }
)
class ESP32PlatformAdapter(Platform):
    def __init__(self, platform_config: dict, platform_settings: dict, event_queue: asyncio.Queue) -> None:
        super().__init__(platform_config, event_queue)
        self.config = platform_config
        self.settings = platform_settings

        # 会话管理
        self.sessions: Dict[str, DeviceSession] = {}
        self.session_lock = asyncio.Lock()

        # WebSocket 服务端
        self.server: Optional[websockets.Server] = None

        # Live2D 实时推送
        self._push_enabled = False
        self._connected_websocket = None  # 当前已连接设备的 websocket

        # 推送日志计数器（每30帧打一次日志）
        self._push_count = 0

        # 确保音频保存目录存在
        os.makedirs(self.config.get("audio_save_dir", "./esp32_audio"), exist_ok=True)

        logger.info("ESP32 适配器 __init__ 完成")

    async def send_by_session(self, session: MessageSesion, message_chain: MessageChain):
        await super().send_by_session(session, message_chain)

    def meta(self) -> PlatformMetadata:
        return PlatformMetadata(
            "esp32",
            "ESP32 智能硬件适配器",
            id='esp32',
        )

    # ------------------------------------------------------------------
    # 模型路径解析（从配置 + 自动发现）
    # ------------------------------------------------------------------
    def _resolve_model_path(self) -> str:
        """
        从配置解析模型路径。
        流程：
        1. 从插件配置读取 `live2d_models_dir`
        2. 从适配器配置读取 `live2d_model_name`
        3. 扫描该目录下的所有子目录，发现可用模型
        4. 如果 `live2d_model_name` 已配置且存在，选中该模型
        5. 否则自动选择第一个发现的模型
        Returns:
            模型文件完整路径（str）。
        Raises:
            FileNotFoundError: 未发现任何模型时抛出。
        """
        from .main import get_esp32_plugin_config
    
        # 👇 插件配置（_conf_schema.json），共享给所有适配器
        plugin_config = get_esp32_plugin_config()
        models_dir = plugin_config.get("live2d_models_dir", "/AstrBot/data/live2d_models")
    
        # 👇 适配器自身配置（cmd_config.json → platform_config），每个适配器独立
        model_name = self.config.get("live2d_model_name", "")
    
        # 扫描可用模型
        available = Live2DService.discover_models(models_dir)
        if not available:
            raise FileNotFoundError(
                f"在 {models_dir} 下未发现任何 Live2D 模型。"
                f"请将模型文件夹放入该目录（每个子目录需包含 model.json 或 *.model3.json）。"
            )
    
        # 如果配置了模型名称
        if model_name:
            if model_name in available:
                logger.info("使用配置的模型: %s → %s", model_name, available[model_name])
                return available[model_name]
            else:
                logger.warning(
                    "配置的模型 '%s' 未找到。可用模型: %s。将自动选择第一个。",
                    model_name, list(available.keys())
                )
    
        # 自动选择第一个
        first_name = list(available.keys())[0]
        first_path = available[first_name]
        logger.info("自动选择模型: [%s] → %s", first_name, first_path)
        return first_path

    # ------------------------------------------------------------------
    # Live2D 实时推送
    # ------------------------------------------------------------------
    async def _push_live2d_frame(self, jpeg_bytes: bytes):
        if not jpeg_bytes or not self._connected_websocket:
            return
        try:
            header = struct.pack(
                '<BBH',
                FRAME_TYPE_LIVE2D_FRAME,
                0x00,
                len(jpeg_bytes)
            )
            await self._connected_websocket.send(header + jpeg_bytes)
            self._push_count += 1
            if self._push_count % 30 == 0:
                logger.info(
                    "Live2D 帧已推送: %d bytes (push #%d)",
                    len(jpeg_bytes), self._push_count
                )
        except Exception as e:
            logger.info("Live2D 推送失败: %s", e)

    def _enable_live2d_push(self):
        if self._push_enabled:
            return
        try:
            # 从配置解析模型路径，初始化 Live2D 服务
            model_path = self._resolve_model_path()
            init_global_service(model_path)

            l2d_service = get_global_service()
            l2d_service.set_push_handler(self._push_live2d_frame)
            self._push_enabled = True
            self._push_count = 0
            logger.info("Live2D 实时推送已启用")
            # 启动后台自动随机动作
            asyncio.create_task(self._auto_motion_task())
        except FileNotFoundError as e:
            logger.error("启用 Live2D 推送失败: %s", e)
        except Exception as e:
            logger.warning("启用 Live2D 推送失败: %s", e)

    async def _auto_motion_task(self):
        """
        后台任务：周期触发随机动作，从模型自动读取可用动作列表。
        """
        first_time = True
        while self._push_enabled:
            try:
                await asyncio.sleep(random.randint(15, 20))
                if not self._push_enabled:
                    break

                l2d_service = get_global_service()
                available = l2d_service.get_available_motions()
                idle_motion = l2d_service.get_idle_motion_name()

                # 过滤掉空闲类动作，只选"主动"动作
                idle_keywords = ['idle', 'Idle', 'breath', 'Breath', 'wait', 'Wait']
                active_motions = [
                    m for m in available
                    if not any(k in m for k in idle_keywords)
                ]
                if not active_motions:
                    active_motions = available  # 兜底

                if first_time and active_motions:
                    motion = active_motions[0]
                    first_time = False
                    logger.info("[AutoMotion] 首次自动触发: %s (验证动作系统)", motion)
                elif active_motions:
                    motion = random.choice(active_motions)
                    logger.info("[AutoMotion] 自动触发随机动作: %s", motion)
                else:
                    motion = idle_motion

                l2d_service.start_motion(motion)
                await asyncio.sleep(7)
                if self._push_enabled:
                    l2d_service.start_motion(idle_motion)
                    logger.info("[AutoMotion] 恢复空闲动作: %s", idle_motion)

            except Exception as e:
                logger.info("[AutoMotion] 异常: %s", e)
                await asyncio.sleep(3)

    def _disable_live2d_push(self):
        if not self._push_enabled:
            return
        try:
            l2d_service = get_global_service()
            l2d_service.set_push_handler(None)
        except Exception:
            pass
        self._push_enabled = False
        self._connected_websocket = None
        logger.info("Live2D 实时推送已禁用")

    # ------------------------------------------------------------------
    # WebSocket 服务器
    # ------------------------------------------------------------------
    async def run(self):
        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("ws_port", 8765)
        auth_token = self.config.get("auth_token", "")

        logger.info("ESP32 适配器启动，监听 %s:%d", host, port)

        async def handler(websocket: WebSocketServerProtocol):
            device_id = None
            session = None
            try:
                if hasattr(websocket, 'request_headers'):
                    headers = websocket.request_headers
                elif hasattr(websocket, 'request'):
                    headers = websocket.request.headers
                else:
                    headers = {}

                # 处理 Authorization 多值头
                try:
                    raw_auth = headers.get("Authorization", "")
                except Exception:
                    raw_auth = ""
                if hasattr(headers, 'get_all'):
                    try:
                        raw_auth = headers.get_all("Authorization", [""])[-1]
                    except Exception:
                        pass
                elif isinstance(headers, dict):
                    raw_auth = headers.get("Authorization", "")

                token = raw_auth.replace("Bearer ", "")

                if auth_token and token != auth_token:
                    logger.warning("认证失败，无效 token: %s", token)
                    await websocket.close(1008, "Invalid token")
                    return

                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("等待 hello 消息超时")
                    await websocket.close(1008, "Hello timeout")
                    return

                try:
                    hello_data = json.loads(raw_msg)
                    if hello_data.get("type") != "hello":
                        raise ValueError("第一条消息必须是 hello")
                    device_id = hello_data.get("device_id")
                    if not device_id:
                        raise ValueError("缺少 device_id")
                    features = hello_data.get("features", {})
                    audio_params = hello_data.get("audio_params", {})
                except Exception as e:
                    logger.error("解析 hello 消息失败: %s", e)
                    await websocket.close(1007, "Invalid hello message")
                    return

                session = DeviceSession(websocket, device_id)
                session.audio_params = audio_params

                async with self.session_lock:
                    if device_id in self.sessions:
                        old_session = self.sessions[device_id]
                        try:
                            await old_session.websocket.close(1000, "New connection")
                        except:
                            pass
                    self.sessions[device_id] = session

                self._connected_websocket = websocket
                self._enable_live2d_push()

                logger.info("设备 %s 已连接，支持功能: %s", device_id, features)

                async for message in websocket:
                    if isinstance(message, bytes):
                        await self._handle_binary_frame(session, message)
                    else:
                        await self._handle_json_message(session, message)

            except websockets.exceptions.ConnectionClosed:
                logger.info("设备 %s 连接已关闭", device_id)
            except Exception as e:
                logger.error("设备 %s 处理异常: %s", device_id, e, exc_info=True)
            finally:
                if self._connected_websocket == websocket:
                    self._disable_live2d_push()
                if device_id:
                    async with self.session_lock:
                        if device_id in self.sessions:
                            del self.sessions[device_id]
                if session and session.is_receiving_audio:
                    await self._finalize_audio_message(session)

        self.server = await websockets.serve(handler, host, port)
        asyncio.create_task(self._check_audio_timeout())
        await self.server.wait_closed()

    # ------------------------------------------------------------------
    # 消息处理
    # ------------------------------------------------------------------
    async def _handle_json_message(self, session: DeviceSession, message: str):
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "start_listening":
                async with self.session_lock:
                    session.audio_buffer.clear()
                    session.is_receiving_audio = True
                    session.last_audio_time = time.time()
                    session.audio_start_time = time.time()
                logger.debug("设备 %s 开始上传音频", session.device_id)

            elif msg_type == "stop_listening":
                await self._finalize_audio_message(session)

            elif msg_type == "text":
                text_content = data.get("content", "")
                if text_content:
                    await self._handle_text_message(session, text_content)

            elif msg_type == "ping":
                await session.websocket.send(json.dumps({"type": "pong"}))

            else:
                logger.warning("未知的消息类型: %s", msg_type)

        except json.JSONDecodeError:
            logger.error("无效的 JSON 消息: %s", message)

    async def _handle_text_message(self, session: DeviceSession, text: str):
        logger.info(
            "[注入诊断] live2d_injection_enabled=%s, text='%s'",
            self.config.get('live2d_injection_enabled', True),
            text[:60]
        )

        if self.config.get("live2d_injection_enabled", True):
            L2D_MARKER = "【L2D_INJECTED】"
            try:
                from .main import get_esp32_context
                ctx = get_esp32_context()
                logger.info("[注入诊断] ctx=%s", ctx)
                if ctx is None:
                    logger.warning("无法获取 ESP32 Context，跳过 Live2D 注入")
                    raise RuntimeError("context is None")

                conv_mgr = ctx.conversation_manager
                if conv_mgr is None:
                    logger.warning("conv_mgr is None，跳过注入")
                    raise RuntimeError("conv_mgr is None")

                uid = "esp32:FriendMessage:%s" % session.device_id
                curr_cid = await conv_mgr.get_curr_conversation_id(uid)
                logger.info("[注入诊断] uid=%s, curr_cid=%s", uid, curr_cid)

                if not curr_cid:
                    try:
                        curr_cid = await conv_mgr.new_conversation(uid, "esp32")
                        logger.info("为设备 %s 创建了新会话: %s", session.device_id, curr_cid)
                    except Exception as e:
                        logger.warning("创建会话失败: %s", e)

                if curr_cid:
                    conv = await conv_mgr.get_conversation(uid, curr_cid)
                    if conv and hasattr(conv, 'history'):
                        if isinstance(conv.history, str):
                            try:
                                history_list = json.loads(conv.history)
                            except (json.JSONDecodeError, TypeError):
                                history_list = []
                        else:
                            history_list = conv.history

                        if isinstance(history_list, list):
                            already_injected = any(
                                L2D_MARKER in str(
                                    msg.get("content", "") if isinstance(msg, dict) else getattr(msg, 'content', '')
                                )
                                for msg in history_list
                            )
                            logger.info(
                                "[注入诊断] already_injected=%s, history_len=%d",
                                already_injected, len(history_list)
                            )

                            if not already_injected:
                                prompt_template = self.config.get(
                                    "live2d_injection_prompt",
                                    "【L2D_INJECTED】\n你是一个搭载了 Live2D 虚拟形象的 AI 助手。当你想通过动作或表情表达情绪时，请在回复文本中插入以下标签：\n  <motion=动作名称>  例如 <motion=TapBody>\n  <expression=表情名称>  例如 <expression=happy>\n可用动作: Idle, TapBody\n可用表情: happy, sad, angry\n不要单独发送这些标签，请将它们自然地嵌入到回复文本中。如果没有合适的动作或表情，可以不添加标签。"
                                )

                                # 动态获取可用的动作和表情
                                l2d_service = get_global_service()
                                avail_motions = l2d_service.get_available_motions()
                                avail_expressions = l2d_service.get_available_expressions()
                                motions_str = ", ".join(avail_motions) if avail_motions else "无可用动作"
                                expressions_str = ", ".join(avail_expressions) if avail_expressions else "无可用表情"

                                l2d_instruction = prompt_template.replace(
                                    "可用动作: Idle, TapBody",
                                    "可用动作: %s" % motions_str
                                ).replace(
                                    "可用表情: happy, sad, angry",
                                    "可用表情: %s" % expressions_str
                                )

                                if not l2d_instruction.startswith(L2D_MARKER):
                                    l2d_instruction = L2D_MARKER + "\n" + l2d_instruction

                                history_list.insert(0, {
                                    "role": "system",
                                    "content": l2d_instruction
                                })

                                await conv_mgr.update_conversation(
                                    uid, curr_cid, history=history_list
                                )
                                logger.info(
                                    "已为设备 %s 注入 Live2D 指令 (动作: %s, 表情: %s)",
                                    session.device_id, motions_str, expressions_str
                                )

            except ImportError as e:
                logger.warning("导入 get_esp32_context 失败: %s", e)
            except Exception as e:
                logger.warning("注入 Live2D 指令失败: %s", e, exc_info=True)

        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.message_str = text
        abm.sender = MessageMember(
            user_id=session.device_id,
            nickname="ESP32_%s" % session.device_id[:6]
        )
        abm.message = [Plain(text=text)]
        abm.raw_message = {"text": text, "device_id": session.device_id}
        abm.self_id = "astrbot"
        abm.session_id = session.device_id
        abm.message_id = "text_%d_%s" % (int(time.time()), session.device_id)
        abm.group_id = None

        await self._commit_message_event(abm, session)

    async def _handle_binary_frame(self, session: DeviceSession, data: bytes):
        if not session.is_receiving_audio:
            logger.warning("设备 %s 未处于接收状态却发送了音频数据", session.device_id)
            return
        if len(data) < 4:
            logger.warning("二进制帧太短")
            return
        try:
            frame_type, reserved, payload_size = struct.unpack('<BBH', data[:4])
            payload = data[4:4+payload_size]
        except Exception as e:
            logger.error("解析二进制帧失败: %s", e)
            return

        if frame_type == FRAME_TYPE_OPUS_AUDIO:
            async with self.session_lock:
                session.audio_buffer.extend(payload)
                session.last_audio_time = time.time()
                max_duration = self.config.get("max_audio_duration", 60)
                elapsed = time.time() - session.audio_start_time
                if elapsed > max_duration:
                    logger.info("音频达到最大时长 %ds，自动结束", max_duration)
                    session.is_receiving_audio = False
                    audio_data = bytes(session.audio_buffer)
                    session.audio_buffer.clear()
                    if audio_data:
                        await self._commit_audio_message(session, audio_data)
        else:
            logger.warning("未知的二进制帧类型: %d", frame_type)

    async def _finalize_audio_message(self, session: DeviceSession):
        async with self.session_lock:
            if not session.is_receiving_audio:
                return
            session.is_receiving_audio = False
            if len(session.audio_buffer) == 0:
                return
            audio_data = bytes(session.audio_buffer)
            session.audio_buffer.clear()
        await self._commit_audio_message(session, audio_data)

    async def _commit_audio_message(self, session: DeviceSession, audio_data: bytes):
        audio_dir = self.config.get("audio_save_dir", "./esp32_audio")
        timestamp = int(time.time())
        file_ext = "opus" if session.audio_params.get("format") == "opus" else "raw"
        audio_file_path = os.path.join(
            audio_dir, "%s_%d.%s" % (session.device_id, timestamp, file_ext)
        )
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)
        logger.info("保存音频文件: %s, 大小: %d bytes", audio_file_path, len(audio_data))

        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.message_str = "[语音消息]"
        abm.sender = MessageMember(
            user_id=session.device_id,
            nickname="ESP32_%s" % session.device_id[:6]
        )
        abm.message = [Record(file=audio_file_path, url=audio_file_path)]
        abm.raw_message = {"audio_path": audio_file_path, "device_id": session.device_id}
        abm.self_id = "astrbot"
        abm.session_id = session.device_id
        abm.message_id = "audio_%d_%s" % (timestamp, session.device_id)
        abm.group_id = None

        await self._commit_message_event(abm, session)

    async def _commit_message_event(self, abm: AstrBotMessage, session: DeviceSession):
        message_event = ESP32Event(
            message_str=abm.message_str,
            message_obj=abm,
            platform_meta=self.meta(),
            session_id=abm.session_id,
            client=session
        )
        self.commit_event(message_event)
        logger.debug("已提交消息事件: %s", abm.session_id)

    async def convert_message(self, data: dict) -> AstrBotMessage:
        raise NotImplementedError("convert_message not implemented for ESP32 adapter")

    async def _check_audio_timeout(self):
        while True:
            await asyncio.sleep(5)
            timed_out = []
            async with self.session_lock:
                for device_id, session in list(self.sessions.items()):
                    if session.is_receiving_audio:
                        if time.time() - session.last_audio_time > session.audio_timeout:
                            timed_out.append(session)
            for session in timed_out:
                logger.info("设备 %s 音频接收超时，自动结束", session.device_id)
                await self._finalize_audio_message(session)

    async def stop(self):
        self._disable_live2d_push()
        try:
            shutdown_global_service()
        except Exception as e:
            logger.warning("关闭 Live2D 服务失败: %s", e)
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("ESP32 适配器已停止")
