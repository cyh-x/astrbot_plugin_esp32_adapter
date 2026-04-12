import asyncio
import json
import struct
import os
import time
from typing import Dict, Optional

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
FRAME_TYPE_OPUS_AUDIO = 0x00      # 上行：OPUS 音频数据
FRAME_TYPE_TTS_AUDIO = 0x01       # 下行：TTS 音频数据
FRAME_TYPE_IMAGE_CHUNK = 0x02     # 下行：图片分片数据

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
        "max_audio_duration": 60
    }
)
class ESP32PlatformAdapter(Platform):
    def __init__(self, platform_config: dict, platform_settings: dict, event_queue: asyncio.Queue) -> None:
        # ✅ 修正：仅传入 event_queue
        super().__init__(platform_config,event_queue)
        self.config = platform_config
        self.settings = platform_settings

        # 会话管理
        self.sessions: Dict[str, DeviceSession] = {}
        self.session_lock = asyncio.Lock()

        # WebSocket 服务端
        self.server: Optional[websockets.Server] = None

        # 确保音频保存目录存在
        os.makedirs(self.config.get("audio_save_dir", "./esp32_audio"), exist_ok=True)
        logger.info("ESP32 适配器 __init__ 完成")

    async def send_by_session(self, session: MessageSesion, message_chain: MessageChain):
        await super().send_by_session(session, message_chain)

    def meta(self) -> PlatformMetadata:
        return PlatformMetadata(
            "esp32",
            "ESP32 智能硬件适配器",
        )

    async def run(self):
        """启动 WebSocket 服务端"""
        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("ws_port", 8765)
        auth_token = self.config.get("auth_token", "")
        logger.info(f"ESP32 适配器启动，监听 {host}:{port}")

        async def handler(websocket: WebSocketServerProtocol):
            device_id = None
            session = None
            try:
                # 认证与握手
                if hasattr(websocket, 'request_headers'):
                    headers = websocket.request_headers
                elif hasattr(websocket, 'request'):
                    headers = websocket.request.headers
                else:
                    headers = {}
                    logger.warning("无法获取 WebSocket 请求头，跳过认证")
                token = headers.get("Authorization", "").replace("Bearer ", "")
                if auth_token and token != auth_token:
                    logger.warning(f"认证失败，无效 token: {token}")
                    await websocket.close(1008, "Invalid token")
                    return

                # 等待 hello 消息
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
                    logger.error(f"解析 hello 消息失败: {e}")
                    await websocket.close(1007, "Invalid hello message")
                    return

                # 创建会话
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

                logger.info(f"设备 {device_id} 已连接，支持功能: {features}")

                # 消息循环
                async for message in websocket:
                    if isinstance(message, bytes):
                        await self._handle_binary_frame(session, message)
                    else:
                        await self._handle_json_message(session, message)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"设备 {device_id} 连接已关闭")
            except Exception as e:
                logger.error(f"设备 {device_id} 处理异常: {e}", exc_info=True)
            finally:
                if device_id:
                    async with self.session_lock:
                        if device_id in self.sessions:
                            del self.sessions[device_id]
                    if session and session.is_receiving_audio:
                        await self._finalize_audio_message(session)

        # 启动 WebSocket 服务器
        self.server = await websockets.serve(handler, host, port)

        # 后台任务：音频超时检查
        asyncio.create_task(self._check_audio_timeout())

        # 保持运行
        await self.server.wait_closed()

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
                logger.debug(f"设备 {session.device_id} 开始上传音频")

            elif msg_type == "stop_listening":
                await self._finalize_audio_message(session)

            elif msg_type == "text":
                text_content = data.get("content", "")
                if text_content:
                    await self._handle_text_message(session, text_content)

            elif msg_type == "ping":
                await session.websocket.send(json.dumps({"type": "pong"}))

            else:
                logger.warning(f"未知的消息类型: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"无效的 JSON 消息: {message}")

    async def _handle_text_message(self, session: DeviceSession, text: str):
        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.message_str = text
        abm.sender = MessageMember(
            user_id=session.device_id,
            nickname=f"ESP32_{session.device_id[:6]}"
        )
        abm.message = [Plain(text=text)]
        abm.raw_message = {"text": text, "device_id": session.device_id}
        abm.self_id = "astrbot"
        abm.session_id = session.device_id
        abm.message_id = f"text_{int(time.time())}_{session.device_id}"
        abm.group_id = None

        await self._commit_message_event(abm, session)

    async def _handle_binary_frame(self, session: DeviceSession, data: bytes):
        if not session.is_receiving_audio:
            logger.warning(f"设备 {session.device_id} 未处于接收状态却发送了音频数据")
            return

        if len(data) < 4:
            logger.warning("二进制帧太短")
            return

        try:
            frame_type, reserved, payload_size = struct.unpack('<BBH', data[:4])
            payload = data[4:4+payload_size]
        except Exception as e:
            logger.error(f"解析二进制帧失败: {e}")
            return

        if frame_type == FRAME_TYPE_OPUS_AUDIO:
            async with self.session_lock:
                session.audio_buffer.extend(payload)
                session.last_audio_time = time.time()

                max_duration = self.config.get("max_audio_duration", 60)
                elapsed = time.time() - session.audio_start_time
                if elapsed > max_duration:
                    logger.info(f"音频达到最大时长 {max_duration}s，自动结束")
                    session.is_receiving_audio = False
                    audio_data = bytes(session.audio_buffer)
                    session.audio_buffer.clear()
            if 'audio_data' in locals() and audio_data:
                await self._commit_audio_message(session, audio_data)
        else:
            logger.warning(f"未知的二进制帧类型: {frame_type}")

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
        audio_file_path = os.path.join(audio_dir, f"{session.device_id}_{timestamp}.{file_ext}")

        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        logger.info(f"保存音频文件: {audio_file_path}, 大小: {len(audio_data)} bytes")

        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.message_str = "[语音消息]"
        abm.sender = MessageMember(
            user_id=session.device_id,
            nickname=f"ESP32_{session.device_id[:6]}"
        )
        abm.message = [Record(file=audio_file_path, url=audio_file_path)]
        abm.raw_message = {"audio_path": audio_file_path, "device_id": session.device_id}
        abm.self_id = "astrbot"
        abm.session_id = session.device_id
        abm.message_id = f"audio_{timestamp}_{session.device_id}"
        abm.group_id = None

        await self._commit_message_event(abm, session)

    async def _commit_message_event(self, abm: AstrBotMessage, session: DeviceSession):
        """提交事件到队列，使用框架的 commit_event 方法"""
        message_event = ESP32Event(
            message_str=abm.message_str,
            message_obj=abm,
            platform_meta=self.meta(),
            session_id=abm.session_id,
            client=session
        )
        # 使用父类方法，与文档示例保持一致
        self.commit_event(message_event)
        logger.debug(f"已提交消息事件: {abm.session_id}")

    async def convert_message(self, data: dict) -> AstrBotMessage:
        # 必须实现，但本项目主要使用二进制音频流，文本消息已通过 _handle_text_message 直接生成 AstrBotMessage
        # 若框架要求必须实现，提供一个最小实现。
        raise NotImplementedError("convert_message not implemented for ESP32 adapter")

    async def _check_audio_timeout(self):
        """后台任务：检查音频接收超时"""
        while True:
            await asyncio.sleep(5)
            timed_out = []
            async with self.session_lock:
                for device_id, session in list(self.sessions.items()):
                    if session.is_receiving_audio:
                        if time.time() - session.last_audio_time > session.audio_timeout:
                            timed_out.append(session)
            for session in timed_out:
                logger.info(f"设备 {session.device_id} 音频接收超时，自动结束")
                await self._finalize_audio_message(session)

    async def stop(self):
        """停止适配器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("ESP32 适配器已停止")
