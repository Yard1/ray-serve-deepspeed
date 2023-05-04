import logging

LOG_FORMAT = (
    "[%(levelname)s %(asctime)s]{rank} %(filename)s: %(lineno)d  " "%(message)s"
)


def get_logger(name=None, rank=-1, **kwargs):
    logger = logging.getLogger(name)
    level = logging.ERROR if rank > 0 else logging.INFO
    format = LOG_FORMAT.format(rank=f"[Rank {rank}]" if rank > -1 else "")
    logging.basicConfig(level=level, format=format, **kwargs)
    return logger
