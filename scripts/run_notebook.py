"""Execute notebooks in place, in order, saving their outputs into them:  python scripts/run_notebook.py <notebook> [...]

Each notebook runs in its own directory with the Jupyter kernel ``$NOTEBOOK_KERNEL`` (default ``roman``). A notebook that
fails stops the run (later notebooks may need its outputs); its outputs up to the error are saved.
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

for arg in sys.argv[1:]:
    path = os.path.abspath(arg)
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    client = NotebookClient(nb, kernel_name=os.environ.get("NOTEBOOK_KERNEL", "roman"), timeout=None,
                            resources={"metadata": {"path": os.path.dirname(path)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
        print(f"{os.path.basename(path)}: executed in {time.time() - t0:.0f} s", flush=True)
