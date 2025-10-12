import os
import re


def get_last_checkpoint(folder):
    checkpoint_reg = re.compile(r"^checkpoint\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if checkpoint_reg.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(checkpoint_reg.search(x).groups()[0])))
