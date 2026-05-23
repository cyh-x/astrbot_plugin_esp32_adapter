from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

_context: Context = None
_plugin_config: AstrBotConfig = None

def get_esp32_context() -> Context:
    return _context

def get_esp32_plugin_config() -> AstrBotConfig:
    return _plugin_config

@register("ESP32", "cyh-x", "一个简单的 ESP32 插件", "1.0.0")
class ESP32Plugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        global _context, _plugin_config
        _context = context
        _plugin_config = config
        from .esp32_adapter import ESP32PlatformAdapter  # noqa

    async def initialize(self):
        pass

    async def terminate(self):
        pass
