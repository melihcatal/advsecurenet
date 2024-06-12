import socket

import pytest

from advsecurenet.utils.network import find_free_port


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert port > 0 and port < 65536

    # Check if the port is actually free by trying to bind to it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_find_free_port_unique():
    port1 = find_free_port()
    # set port1 to be busy
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port1))
    port2 = find_free_port()
    assert port1 != port2
