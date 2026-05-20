from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

# ---- 全局 Context 模式（供 esp32_adapter 获取 conversation_manager） ----
_context: Context = None

def get_esp32_context() -> Context:
    """获取 ESP32 插件的全局 Context 实例"""
    return _context


@register("ESP32", "cyh-x", "一个简单的 ESP32 插件", "1.0.0")
class ESP32Plugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 保存全局 Context，供 esp32_adapter 中的注入逻辑使用
        global _context
        _context = context
        # 导入适配器以触发注册
        from .esp32_adapter import ESP32PlatformAdapter  # noqa

    async def initialize(self):
        pass

    async def terminate(self):
        pass
