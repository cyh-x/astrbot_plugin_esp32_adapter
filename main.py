from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger

@register("ESP32", "cyh-x", "一个简单的 ESP32 插件", "1.0.0")
class ESP32Plugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        # 导入适配器以触发注册
        from .esp32_adapter import ESP32PlatformAdapter  # noqa

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        pass

    async def terminate(self):
        """可选择实现异步的插件销毁方法，当插件被卸载/停用时会调用。"""
        # 停止适配器服务
        # 可以通过 context 获取适配器实例，此处简化处理
        pass
