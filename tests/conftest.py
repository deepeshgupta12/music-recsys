import pytest
from starlette.testclient import TestClient

from musicrec.api.main import app


@pytest.fixture()
def client() -> TestClient:
    """
    Standard API test client for the FastAPI app.
    Use in tests as: def test_x(client): ...
    """
    return TestClient(app)


@pytest.fixture()
def client_follow_redirects() -> TestClient:
    """
    Same as client, but follows redirects automatically.
    Helpful for endpoints that redirect (if any are added later).
    """
    return TestClient(app, follow_redirects=True)


@pytest.fixture()
def user_id() -> str:
    """
    Handy default user_id fixture for personalization / feedback tests.
    Override in tests if you need a specific value.
    """
    return "user_test_default"