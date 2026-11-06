"""
RouteWise: Notebook Runner

Executes notebooks/04_regression_and_residuals.ipynb end-to-end,
captures all cell outputs, and verifies zero execution errors.
"""

import os
import nbformat
from nbclient import NotebookClient

notebook_path = "notebooks/04_regression_and_residuals.ipynb"

print(f"Reading notebook from: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook cells with nbclient...")
client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": "notebooks"}}
)
client.execute()

print(f"Writing executed notebook back to: {notebook_path}")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully!")
