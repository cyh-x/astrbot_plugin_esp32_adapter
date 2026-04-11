import asyncio
import json
import struct
import os
import tempfile
import time
from typing import Dict, Optional



# 或者直接用 websockets 模块中的类型
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
        self.audio_buffer = bytearray()          # 累积的音频数据
        self.audio_params = {}                   # 音频参数
        self.is_receiving_audio = False          # 是否正在接收音频流
        self.last_audio_time = 0                 # 最后一次收到音频的时间戳
        self.audio_start_time = 0                # 本次音频开始的时间戳（用于最大时长）
        self.audio_timeout = 5.0                 # 音频静音超时自动结束（秒）

@register_platform_adapter(
    "esp32",
    "ESP32 智能硬件适配器",
    default_config_tmpl={
        "host": "0.0.0.0",
        "ws_port": 8765,
        "auth_token": "",           # 可选，用于连接认证
        "audio_save_dir": "./esp32_audio",  # 音频临时保存目录
        "max_audio_duration": 60    # 单次最大录音时长（秒）
    }
)
class ESP32PlatformAdapter(Platform):
    def __init__(self, platform_config: dict, platform_settings: dict, event_queue: asyncio.Queue) -> None:
        super().__init__(platform_config, platform_settings, event_queue)  # 注意顺序！

        self.config = platform_config
        self.settings = platform_settings
        self.event_queue = event_queue
        
        # 其他初始化代码保持不变...
        
        # 会话管理
        self.sessions: Dict[str, DeviceSession] = {}
        self.session_lock = asyncio.Lock()
        
        # WebSocket 服务端
        self.server: Optional[websockets.Server] = None
        
        # 确保音频保存目录存在
        os.makedirs(self.config.get("audio_save_dir", "./esp32_audio"), exist_ok=True)
        logger.debug("ESP32 适配器 __init__ 执行")
    async def send_by_session(self, session: MessageSesion, message_chain: MessageChain):
        """通过会话发送消息（必须实现）"""
        # 调用父类默认实现，内部会找到对应的事件并调用 send 方法
        await super().send_by_session(session, message_chain)

    def meta(self) -> PlatformMetadata:
        """返回平台元数据"""
        return PlatformMetadata(
            "esp32",
            "ESP32 智能硬件适配器",
        )

    async def run(self):
        """启动 WebSocket 服务端"""
        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("ws_port", 8765)
        auth_token = self.config.get("auth_token", "")
        logger.debug("ESP32 适配器 run 方法被调用")
        logger.info(f"ESP32 适配器启动，监听 {host}:{port}")

        async def handler(websocket: WebSocketServerProtocol):
            """处理单个 WebSocket 连接"""
            device_id = None
            session = None
            try:
                # 1. 认证与握手
                # 兼容不同版本 websockets 获取请求头的方式
                if hasattr(websocket, 'request_headers'):
                    headers = websocket.request_headers
                elif hasattr(websocket, 'request'):
                    headers = websocket.request.headers
                else:
                    headers = {}
                    logger.warning("无法获取 WebSocket 请求头，将跳过认证")
                token = headers.get("Authorization", "").replace("Bearer ", "")
                if auth_token and token != auth_token:
                    logger.warning(f"认证失败，无效 token: {token}")
                    await websocket.close(1008, "Invalid token")
                    return

                # 等待客户端发送 hello 消息（JSON）
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
                    # 如果已有同一设备连接，关闭旧连接
                    if device_id in self.sessions:
                        old_session = self.sessions[device_id]
                        try:
                            await old_session.websocket.close(1000, "New connection")
                        except:
                            pass
                    self.sessions[device_id] = session

                logger.info(f"设备 {device_id} 已连接，支持功能: {features}")

                # 2. 进入消息循环
                async for message in websocket:
                    if isinstance(message, bytes):
                        # 二进制帧处理（音频数据）
                        await self._handle_binary_frame(session, message)
                    else:
                        # JSON 控制消息
                        await self._handle_json_message(session, message)

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"设备 {device_id} 连接已关闭")
            except Exception as e:
                logger.error(f"设备 {device_id} 处理异常: {e}", exc_info=True)
            finally:
                # 清理会话
                if device_id:
                    async with self.session_lock:
                        if device_id in self.sessions:
                            del self.sessions[device_id]
                    # 如果正在接收音频且未正常结束，强制结束并提交消息
                    if session and session.is_receiving_audio:
                        await self._finalize_audio_message(session)

        # 启动 WebSocket 服务器
        self.server = await websockets.serve(handler, host, port)
        
        # 启动音频超时检查任务
        asyncio.create_task(self._check_audio_timeout())
        
        # 保持运行
        await self.server.wait_closed()

    async def _handle_json_message(self, session: DeviceSession, message: str):
        """处理 JSON 控制消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "start_listening":
                # 开始接收音频流
                async with self.session_lock:
                    session.audio_buffer.clear()
                    session.is_receiving_audio = True
                    session.last_audio_time = time.time()
                    session.audio_start_time = time.time()
                logger.debug(f"设备 {session.device_id} 开始上传音频")
                
            elif msg_type == "stop_listening":
                # 结束接收音频流，并生成消息事件
                await self._finalize_audio_message(session)
                
            elif msg_type == "text":
                # 处理文本消息
                text_content = data.get("content", "")
                if text_content:
                    await self._handle_text_message(session, text_content)
                
            elif msg_type == "ping":
                # 心跳响应
                await session.websocket.send(json.dumps({"type": "pong"}))
                
            else:
                logger.warning(f"未知的消息类型: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"无效的 JSON 消息: {message}")

    async def _handle_text_message(self, session: DeviceSession, text: str):
        """将文本消息转换为 AstrBot 事件"""
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
        """处理二进制帧（音频数据）"""
        if not session.is_receiving_audio:
            logger.warning(f"设备 {session.device_id} 未处于接收状态却发送了音频数据")
            return

        # 解析自定义二进制协议
        # struct: uint8 type, uint8 reserved, uint16 payload_size, bytes payload
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
                
                # 检查是否超过最大时长（基于时间）
                max_duration = self.config.get("max_audio_duration", 60)
                elapsed = time.time() - session.audio_start_time
                if elapsed > max_duration:
                    logger.info(f"音频达到最大时长 {max_duration}s，自动结束")
                    # 需要释放锁后调用 _finalize_audio_message，但要避免死锁
                    # 先标记为不再接收，稍后在锁外调用
                    session.is_receiving_audio = False
                    # 复制 buffer 并清空
                    audio_data = bytes(session.audio_buffer)
                    session.audio_buffer.clear()
            # 在锁外提交消息
            if frame_type == FRAME_TYPE_OPUS_AUDIO and 'audio_data' in locals() and audio_data:
                await self._commit_audio_message(session, audio_data)
        else:
            logger.warning(f"未知的二进制帧类型: {frame_type}")

    async def _finalize_audio_message(self, session: DeviceSession):
        """完成音频接收，保存文件并创建 AstrBot 消息事件（外部调用入口）"""
        async with self.session_lock:
            if not session.is_receiving_audio:
                return
            session.is_receiving_audio = False
            if len(session.audio_buffer) == 0:
                return
            audio_data = bytes(session.audio_buffer)
            session.audio_buffer.clear()
        # 在锁外执行 I/O 和消息提交
        await self._commit_audio_message(session, audio_data)

    async def _commit_audio_message(self, session: DeviceSession, audio_data: bytes):
        """保存音频文件并提交消息事件（无锁，纯 I/O）"""
        # 保存音频到临时文件
        audio_dir = self.config.get("audio_save_dir", "./esp32_audio")
        timestamp = int(time.time())
        file_ext = "opus" if session.audio_params.get("format") == "opus" else "raw"
        audio_file_path = os.path.join(audio_dir, f"{session.device_id}_{timestamp}.{file_ext}")
        
        # 异步写文件（为避免阻塞事件循环，可以使用 aiofiles，此处暂用普通 write）
        # 如果音频数据较大，建议使用 aiofiles.open
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"保存音频文件: {audio_file_path}, 大小: {len(audio_data)} bytes")
        
        # 构造 AstrBotMessage
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
        """将 AstrBotMessage 提交到事件队列"""
        message_event = ESP32Event(
            message_str=abm.message_str,
            message_obj=abm,
            platform_meta=self.meta(),
            session_id=abm.session_id,
            client=session
        )
        await self.event_queue.put(message_event)
        logger.debug(f"已提交消息事件: {abm.session_id}")

    async def convert_audio_message(self, session: DeviceSession, audio_path: str) -> AstrBotMessage:
        """将音频文件转换为 AstrBot 消息（保留接口兼容性）"""
        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.message_str = "[语音消息]"
        abm.sender = MessageMember(
            user_id=session.device_id,
            nickname=f"ESP32_{session.device_id[:6]}"
        )
        abm.message = [Record(file=audio_path, url=audio_path)]
        abm.raw_message = {"audio_path": audio_path, "device_id": session.device_id}
        abm.self_id = "astrbot"
        abm.session_id = session.device_id
        abm.message_id = f"audio_{int(time.time())}_{session.device_id}"
        abm.group_id = None
        return abm

    async def convert_message(self, data: dict) -> AstrBotMessage:
        """转换通用消息（适配器要求实现，但本例中不使用）"""
        # 如有需要可扩展
        raise NotImplementedError("convert_message not implemented for ESP32 adapter")

    async def handle_msg(self, message: AstrBotMessage, session: DeviceSession):
        """处理消息并提交事件队列（保留兼容性）"""
        await self._commit_message_event(message, session)

    async def _check_audio_timeout(self):
        """后台任务：检查音频接收超时（无锁内 await）"""
        while True:
            await asyncio.sleep(5)
            # 收集超时的会话
            timed_out = []
            async with self.session_lock:
                for device_id, session in list(self.sessions.items()):
                    if session.is_receiving_audio:
                        if time.time() - session.last_audio_time > session.audio_timeout:
                            timed_out.append(session)
            # 在锁外处理超时
            for session in timed_out:
                logger.info(f"设备 {session.device_id} 音频接收超时，自动结束")
                await self._finalize_audio_message(session)

    async def stop(self):
        """停止适配器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("ESP32 适配器已停止")
