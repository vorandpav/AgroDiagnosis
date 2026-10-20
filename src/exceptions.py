"""Exceptions for uploaded test dataset evaluation."""


class InvalidDatasetError(ValueError):
    """Reject an invalid archive or image."""


class EvaluationTrackingError(RuntimeError):
    """Report unavailable evaluation tracking."""
