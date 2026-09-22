class TraceGuardError(RuntimeError):
    """Base error with a user-actionable message."""


class ConfigurationError(TraceGuardError):
    """Raised when a configuration is invalid."""


class DependencyError(TraceGuardError):
    """Raised when an optional upstream dependency is unavailable."""


class WeightNotFoundError(TraceGuardError):
    """Raised when an explicitly configured checkpoint is missing."""

