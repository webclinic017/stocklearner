import logging


def get_file_logger(logger_name, log_file_path, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=level)

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
