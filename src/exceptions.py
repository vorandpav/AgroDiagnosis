"""Exceptions for uploaded test dataset evaluation and task publishing."""


class InvalidDatasetError(ValueError):
    """Reject an invalid archive or image."""


class EvaluationTrackingError(RuntimeError):
    """Report unavailable evaluation tracking."""


class QueueFullError(Exception):
    """Raised when RabbitMQ rejects a publish because the task queue is full."""
