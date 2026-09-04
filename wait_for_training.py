"""Block until no training process is running.

Sequencing on `pgrep -f <script>` deadlocked earlier: a waiter's own command
line contains the pattern it searches for, so waiters matched each other and
each other's shells.  This checks for the actual work instead -- a python
process whose argv contains train.py -- which no shell wrapper can imitate.
"""

import glob
import os
import sys
import time


def training_running():
    for path in glob.glob('/proc/*/cmdline'):
        try:
            parts = [p for p in open(path, 'rb').read().decode(errors='ignore').split('\x00') if p]
        except (OSError, IOError):
            continue
        if not parts or 'python' not in os.path.basename(parts[0]):
            continue
        if any(p.endswith('train.py') for p in parts):
            return True
    return False


if __name__ == "__main__":
    while training_running():
        time.sleep(30)
    sys.exit(0)
