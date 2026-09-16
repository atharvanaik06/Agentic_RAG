"""Package smoke-test entry point."""

from advanced_rag import __version__
from advanced_rag.config import get_settings
from advanced_rag.logging import configure_logging, get_logger


def main() -> None:
    """Load the application settings and emit a safe startup message."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger = get_logger(__name__)
    logger.info(
        "application_ready",
        extra={
            "app_name": settings.app_name,
            "environment": settings.environment,
            "version": __version__,
        },
    )


if __name__ == "__main__":
    main()
