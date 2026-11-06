"""
Helper module to build and execute Jupyter notebooks with embedded outputs and figures.
"""

import ast
import base64
import contextlib
import io
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import nbformat as nbf


class NotebookExecutor:
    def __init__(self):
        self.scope = {}
        self.execution_count = 0

    def add_markdown(self, nb, text):
        nb.cells.append(nbf.v4.new_markdown_cell(text))

    def add_code(self, nb, code_text):
        self.execution_count += 1
        cell = nbf.v4.new_code_cell(code_text)
        cell.execution_count = self.execution_count

        stdout_buf = io.StringIO()
        plt.close("all")  # reset any previous plots

        # Split into statements and final expression if present
        try:
            tree = ast.parse(code_text)
        except Exception as e:
            stdout_buf.write(f"SyntaxError: {e}\n")
            cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=stdout_buf.getvalue()))
            nb.cells.append(cell)
            return

        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()  # remove last expression to evaluate it separately
            exec_code = compile(tree, filename="<cell>", mode="exec")
            eval_code = compile(ast.Expression(body=last_expr.value), filename="<cell>", mode="eval")
        else:
            exec_code = compile(tree, filename="<cell>", mode="exec")
            eval_code = None

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            try:
                exec(exec_code, self.scope)
                eval_val = eval(eval_code, self.scope) if eval_code else None
            except Exception as e:
                import traceback
                traceback.print_exc(file=stdout_buf)
                eval_val = None

        outputs = []
        stdout_str = stdout_buf.getvalue()
        if stdout_str:
            outputs.append(nbf.v4.new_output("stream", name="stdout", text=stdout_str))

        # Check if any matplotlib figures were generated
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            img_buf.seek(0)
            b64_data = base64.b64encode(img_buf.read()).decode("utf-8")
            outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": b64_data,
                    "text/plain": "<Figure size ...>"
                }
            ))
        plt.close("all")

        # Format eval_val (e.g. DataFrame)
        if eval_val is not None:
            data = {"text/plain": repr(eval_val)}
            if hasattr(eval_val, "_repr_html_"):
                data["text/html"] = eval_val._repr_html_()
            outputs.append(nbf.v4.new_output(
                output_type="execute_result",
                data=data,
                execution_count=self.execution_count
            ))

        cell.outputs = outputs
        nb.cells.append(cell)
