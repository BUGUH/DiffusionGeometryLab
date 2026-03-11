from src.schedulers.base_scheduler import BaseScheduler
from src.schedulers.vp_scheduler import VPScheduler
from src.schedulers.ve_scheduler import VEScheduler
from src.schedulers.of_scheduler import OFScheduler


def build_scheduler(cfg) -> BaseScheduler:
    """根据配置构建调度器。"""
    stype = cfg.scheduler.type.lower()
    if stype == "vp":
        return VPScheduler(cfg)
    elif stype == "ve":
        return VEScheduler(cfg)
    elif stype == "of":
        return OFScheduler(cfg)
    else:
        raise ValueError(f"未知调度器类型: {stype}")


__all__ = ["BaseScheduler", "VPScheduler", "VEScheduler", "OFScheduler", "build_scheduler"]