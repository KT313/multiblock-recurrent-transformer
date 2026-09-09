# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Network helpers for the tests that open a process group (`training/backend/test_ddp.py`, `training/test_distributed.py`).
"""

import socket


def free_port() -> int:
    """
    A TCP port that was free on 127.0.0.1 a moment ago (the OS picked it), for a rendezvous in a test.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
