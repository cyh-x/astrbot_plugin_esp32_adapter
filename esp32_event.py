import os
import struct
import json
import asyncio
from typing import TYPE_CHECKING

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.platform import AstrBotMessage, PlatformMetadata
from astrbot.api.message_components import Plain, Image, Record
from astrbot.core.utils.io import download_image_by_url
from astrbot import logger
import time
from PIL import Image as PILImage
if TYPE_CHECKING:
    from .esp32_adapter import DeviceSession

# 二进制帧类型常量（与适配器保持一致）
FRAME_TYPE_TTS_AUDIO = 0x01
FRAME_TYPE_IMAGE_CHUNK = 0x02

# 图片分片大小（字节），可根据 ESP32 内存调整
IMAGE_CHUNK_SIZE = 1024 * 8  # 8KB


class ESP32Event(AstrMessageEvent):
    def __init__(
        self,
        message_str: str,
        message_obj: AstrBotMessage,
        platform_meta: PlatformMetadata,
        session_id: str,
        client: 'DeviceSession'
    ):
        super().__init__(message_str, message_obj, platform_meta, session_id)
        self.client = client
        self.websocket = client.websocket

    async def send(self, message: MessageChain):
        """
        将 AstrBot 回复的消息链发送给 ESP32 设备
        支持 Plain（文本）、Image（图片）、Record（音频）组件
        """
        for component in message.chain:
            if isinstance(component, Plain):
                await self._send_text(component.text)
                
            elif isinstance(component, Image):
                await self._send_image(component)
                
            elif isinstance(component, Record):
                if component.text:
                    await self._send_text(component.text)
                await self._send_audio(component)
                
            else:
                logger.warning(f"ESP32 不支持的消息组件类型: {type(component)}")

        # 必须调用父类方法
        await super().send(message)

    async def _send_text(self, text: str):
        """发送文本消息（通过 JSON）"""
        if not text:
            return
        msg = {
            "type": "text",
            "content": text
        }
        try:
            await self.websocket.send(json.dumps(msg))
            logger.debug(f"发送文本: {text[:50]}...")
        except Exception as e:
            logger.error(f"发送文本失败: {e}")

    async def _send_image(self, image: Image):
        """发送图片，分片传输（先缩放到 320x240，减少传输量）"""
        # 获取图片本地路径
        img_path = await self._resolve_image_path(image)
        if not img_path or not os.path.exists(img_path):
            logger.error(f"图片文件不存在: {img_path}")
            return

        try:
            # ★ 用 Pillow 缩放到 ESP32 屏幕分辨率
            with PILImage.open(img_path) as pil_img:
                max_w, max_h = 320, 240
                pil_img.thumbnail((max_w, max_h), PILImage.LANCZOS)
                temp_dir = os.path.dirname(img_path) or "/AstrBot/data/temp"
                temp_path = os.path.join(temp_dir, f".esp32_resized_{int(time.time())}.jpg")
                pil_img.save(temp_path, "JPEG", quality=85)

            file_size = os.path.getsize(temp_path)
            total_chunks = (file_size + IMAGE_CHUNK_SIZE - 1) // IMAGE_CHUNK_SIZE

            # 先发送图片元信息
            meta_msg = {
                "type": "image_start",
                "total_chunks": total_chunks,
                "file_size": file_size
            }
            await self.websocket.send(json.dumps(meta_msg))

            # 分片发送缩放后的图片数据
            with open(temp_path, "rb") as f:
                chunk_index = 0
                while True:
                    chunk_data = f.read(IMAGE_CHUNK_SIZE)
                    if not chunk_data:
                        break
                    flags = 0x01 if (chunk_index == total_chunks - 1) else 0x00
                    header = struct.pack('<BBH', FRAME_TYPE_IMAGE_CHUNK, flags, len(chunk_data))
                    await self.websocket.send(header + chunk_data)
                    chunk_index += 1
                    await asyncio.sleep(0.01)

            # 清理临时文件
            try:
                os.remove(temp_path)
            except:
                pass

            logger.info(f"图片发送完成: {img_path} -> 缩放后 {temp_path}, "
                        f"分片数: {total_chunks}, 大小: {file_size} bytes")
        except Exception as e:
            logger.error(f"发送图片失败: {e}")
    async def _send_audio(self, record: Record):
        """发送音频（TTS 结果）"""
        audio_path = await self._resolve_audio_path(record)
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return

        try:
            file_size = os.path.getsize(audio_path)
            
            # 发送音频开始通知
            meta_msg = {
                "type": "tts_start",
                "file_size": file_size
            }
            await self.websocket.send(json.dumps(meta_msg))

            # 分块发送音频数据（假设每次发送 4KB）
            chunk_size = 4096
            with open(audio_path, "rb") as f:
                while True:
                    chunk_data = f.read(chunk_size)
                    if not chunk_data:
                        break
                    
                    # 构造音频帧头
                    header = struct.pack(
                        '<BBH',
                        FRAME_TYPE_TTS_AUDIO,
                        0x00,  # 保留
                        len(chunk_data)
                    )
                    await self.websocket.send(header + chunk_data)
                    await asyncio.sleep(0.01)
            
            # 发送结束标记（空 payload 表示结束）
            end_header = struct.pack('<BBH', FRAME_TYPE_TTS_AUDIO, 0x01, 0)
            await self.websocket.send(end_header)
            
            logger.info(f"音频发送完成: {audio_path}")
            
        except Exception as e:
            logger.error(f"发送音频失败: {e}")

    async def _resolve_image_path(self, image: Image) -> str:
        """解析图片组件的本地路径"""
        if image.file:
            # 如果是本地文件路径
            if image.file.startswith("file://"):
                return image.file[7:]
            elif image.file.startswith("http"):
                # 下载网络图片
                return await download_image_by_url(image.file)
            else:
                return image.file
        elif image.url:
            if image.url.startswith("http"):
                return await download_image_by_url(image.url)
            else:
                return image.url
        return ""

    async def _resolve_audio_path(self, record: Record) -> str:
        """解析音频组件的本地路径"""
        if record.file:
            if record.file.startswith("file://"):
                return record.file[7:]
            return record.file
        elif record.url:
            if record.url.startswith("http"):
                # 如有需要可下载
                return await download_image_by_url(record.url)  # 复用下载函数
            return record.url
        return ""
