"""Application logging configuration."""

import json
import logging
from datetime import UTC, datetime
from typing import Literal


class JsonFormatter(logging.Formatter):
    """Render standard log records as one JSON object per line."""

    _standard_attributes = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self._standard_attributes and key not in {"message", "asctime"}
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO", log_format: Literal["console", "json"] = "console"
) -> None:
    """Configure root logging consistently for every application entry point."""
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a named standard-library logger."""
    return logging.getLogger(name)
