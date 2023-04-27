from typing import Type

from ._base import BasePipeline
from .dolly2_pipeline import DollyV2Pipeline
from .stablelm_pipeline import StableLMPipeline


def get_pipeline_cls_by_name(name: str) -> Type[BasePipeline]:
    lowercase_globals = {k.lower(): v for k, v in globals().items()}
    ret = lowercase_globals.get(
        f"{name.lower()}pipeline", lowercase_globals.get(name.lower(), None)
    )
    assert ret
    return ret


__all__ = ["get_pipeline_cls_by_name", "DollyV2Pipeline", "StableLMPipeline"]
