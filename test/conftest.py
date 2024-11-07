import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--api-token",
        action="store",
        default=None,
        help="API token for authentication",
    )


@pytest.fixture(autouse=True)
def skip_if_no_api_token(request):
    if request.node.get_closest_marker("requires_api_token"):
        token = request.config.getoption("--api-token")
        if not token:
            pytest.skip("Skipping test: API token is not provided.")


@pytest.fixture(scope="module")
def setup_module(request):
    api_token = request.config.getoption("--api-token")

    # Setup code
    if api_token:
        print(f"\nSetup: Initializing resources with API token: {api_token}")
    else:
        print("\nSetup: No API token provided. Some tests will be skipped.")

    yield api_token

    # Teardown code
    print(f"\nTeardown: Cleaning up resources initialized with API token: {api_token}")
