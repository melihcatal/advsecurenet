import socket
from advsecurenet.utils.network import find_free_port


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int), "The function should return an integer."
    assert 1024 <= port <= 65535, "Port number should be in the valid range (1024-65535)."

    # Test if the port is actually free
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
        except OSError:
            pytest.fail(f"Port {port} is not free.")
