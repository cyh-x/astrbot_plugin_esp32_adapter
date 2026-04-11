import struct
from enum import IntEnum

PROTOCOL_VERSION = 3

class FrameType(IntEnum):
    AUDIO_OPUS = 0x00
    IMAGE_CHUNK = 0x01
    TTS_AUDIO = 0x02

def encode_binary_audio_frame(payload: bytes, is_tts: bool = False) -> bytes:
    """编码音频帧：type(1) + reserved(1) + payload_len(2) + payload"""
    frame_type = FrameType.TTS_AUDIO if is_tts else FrameType.AUDIO_OPUS
    header = struct.pack(">BBH", frame_type, 0, len(payload))
    return header + payload

def encode_binary_image_chunk(payload: bytes, chunk_index: int, is_last: bool) -> bytes:
    """编码图片分片：type(1) + flags(1) + chunk_index(2) + payload_len(2) + payload"""
    flags = 0x01 if is_last else 0x00
    header = struct.pack(">BBHH", FrameType.IMAGE_CHUNK, flags, chunk_index, len(payload))
    return header + payload

def decode_binary_frame(data: bytes):
    """解码二进制帧，返回 (frame_type, payload)"""
    if len(data) < 4:
        raise ValueError("帧头不足 4 字节")
    frame_type, reserved, payload_len = struct.unpack(">BBH", data[:4])
    if len(data) < 4 + payload_len:
        raise ValueError("帧数据不完整")
    payload = data[4:4+payload_len]
    return frame_type, payload