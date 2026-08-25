"""Regression tests for model error-retry classification (model_backends._should_retry)."""

from agent_project.model_backends import OpenAIBackend


class _StatusError(Exception):
    def __init__(self, status=None):
        self.status_code = status
        super().__init__("boom")


def test_should_retry_transient_statuses():
    b = OpenAIBackend.__new__(OpenAIBackend)
    for st in (408, 429, 500, 502, 503, 504):
        assert b._should_retry(_StatusError(st)) is True, st


def test_should_retry_permanent_4xx_not_retried():
    b = OpenAIBackend.__new__(OpenAIBackend)
    # 401/403 auth failures and other permanent 4xx must NOT be retried.
    for st in (400, 401, 403, 404, 405, 406, 410, 422):
        assert b._should_retry(_StatusError(st)) is False, st


def test_should_retry_network_errors():
    b = OpenAIBackend.__new__(OpenAIBackend)
    assert b._should_retry(ConnectionError("down")) is True
    assert b._should_retry(TimeoutError("slow")) is True


def test_should_retry_unclassified_is_transient():
    b = OpenAIBackend.__new__(OpenAIBackend)
    # status is None (connection-level failure) -> treat as transient
    assert b._should_retry(_StatusError(None)) is True
