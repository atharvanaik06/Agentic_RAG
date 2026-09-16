import json
import logging

from advanced_rag.logging import JsonFormatter, configure_logging


def test_json_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord(
        name="advanced_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="retrieval_complete",
        args=(),
        exc_info=None,
    )
    record.chunk_count = 4

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "retrieval_complete"
    assert payload["chunk_count"] == 4
    assert payload["level"] == "INFO"


def test_configure_logging_replaces_root_handlers() -> None:
    configure_logging(level="WARNING", log_format="console")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) == 1
