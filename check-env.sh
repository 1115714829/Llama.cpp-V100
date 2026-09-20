#!/bin/bash
echo "=== sudo -n (passwordless?) ==="
if sudo -n true 2>/dev/null; then echo SUDO_NOPASSWD; else echo SUDO_NEEDS_PASSWORD; fi
echo "=== python3 / pip ==="
command -v python3 && python3 --version 2>&1
(command -v pip3 && pip3 --version) 2>/dev/null
(command -v pip && pip --version) 2>/dev/null
echo "=== paramiko ==="
python3 -c "import paramiko; print('paramiko', paramiko.__version__)" 2>&1 | head -1
echo "=== apt-get ==="
command -v apt-get && echo apt_get_present
echo "=== network to pypi ==="
python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=8); print('NET_OK')" 2>&1 | head -1
echo CHECK_DONE
