def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to run the tests on. Default is 'cpu'.",
    )
