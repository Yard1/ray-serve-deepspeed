from typing import Type

from ._base import LLMInitializer
from .deepspeed import DeepSpeedInitializer
from .device_map import DeviceMapInitializer
from .single_device import SingleDeviceInitializer


def get_initializer_cls_by_name(name: str) -> Type[LLMInitializer]:
    lowercase_globals = {k.lower(): v for k, v in globals().items()}
    ret = lowercase_globals.get(
        f"{name.lower()}initializer", lowercase_globals.get(name.lower(), None)
    )
    assert ret
    return ret


__all__ = [
    "get_initializer_cls_by_name",
    "DeepSpeedInitializer",
    "DeviceMapInitializer",
    "SingleDeviceInitializer",
]
