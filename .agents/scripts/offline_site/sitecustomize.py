"""发布验证时阻止 Python 进程建立网络连接。"""

import socket


def blocked(*_args, **_kwargs):
    raise RuntimeError('network access is disabled by BDC2026 release verification')


socket.create_connection = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
