"""Run the lab notebooks in IPython and save their outputs."""
from contextlib import chdir
from pathlib import Path

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output


def execute_notebook(path, shell):
    notebook = nbformat.read(path, as_version=4)
    shell.reset(new_session=False)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    with chdir(path.parent):
        for count, cell in enumerate(code_cells, start=1):
            with capture_output(display=True) as captured:
                # Save the final expression as a rich notebook output below.
                original_quiet = shell.displayhook.quiet
                shell.displayhook.quiet = lambda: True
                try:
                    result = shell.run_cell(cell.source, store_history=False)
                finally:
                    shell.displayhook.quiet = original_quiet

            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise RuntimeError(f"{path.name}: code cell {count} failed") from error

            outputs = []
            stdout = captured.stdout.replace("No event loop hook running.\n", "")
            for name, content in (("stdout", stdout), ("stderr", captured.stderr)):
                if content:
                    outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
            for display in captured.outputs:
                outputs.append(nbformat.v4.new_output(
                    "display_data", data=display.data, metadata=display.metadata,
                ))
            if result.result is not None:
                data, metadata = shell.display_formatter.format(result.result)
                outputs.append(nbformat.v4.new_output(
                    "execute_result", execution_count=count, data=data, metadata=metadata,
                ))
            cell.execution_count = count
            cell.outputs = outputs

    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    project_dir = Path(__file__).resolve().parent
    notebooks = sorted((project_dir / "labs").rglob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No lab notebooks were found.")

    shell = TerminalInteractiveShell.instance(simple_prompt=True)
    shell.display_formatter.active_types = ["text/plain", "text/html", "image/png", "image/svg+xml"]
    for path in notebooks:
        print(f"Running {path.relative_to(project_dir)}", flush=True)
        execute_notebook(path, shell)
    print(f"Completed {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
