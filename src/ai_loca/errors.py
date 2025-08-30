class AIAdapterError(Exception):
    """Base exception for adapter-related failures."""


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


class HTTPRequestError(Exception):
    """Raised when an HTTP request fails irrecoverably."""


class FFmpegError(Exception):
    """Raised when FFmpeg invocation fails or returns non-zero."""


class OrchestratorError(Exception):
    """Raised when the orchestration pipeline fails at a high level."""

