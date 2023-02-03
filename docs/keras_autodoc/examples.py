import os
import pathlib


def copy_examples(examples_dir, destination_dir):
    """Copy the examples directory in the documentation.

    Prettify files by extracting the docstrings written in Markdown.
    """
    pathlib.Path(destination_dir).mkdir(exist_ok=True)
    for file in os.listdir(examples_dir):
        if not file.endswith(".py"):
            continue
        module_path = os.path.join(examples_dir, file)
        docstring, starting_line = get_module_docstring(module_path)
        destination_file = os.path.join(destination_dir, file[:-2] + "md")
        with open(destination_file, "w+", encoding="utf-8") as f_out, open(
            examples_dir / file, "r+", encoding="utf-8"
        ) as f_in:
            f_out.write(docstring + "\n\n")

            # skip docstring
            for _ in range(starting_line):
                next(f_in)

            f_out.write("```python\n")
            # next line might be empty.
            line = next(f_in)
            if line != "\n":
                f_out.write(line)

            # copy the rest of the file.
            for line in f_in:
                f_out.write(line)
            f_out.write("```")


def get_module_docstring(filepath):
    """Extract the module docstring.

    Also finds the line at which the docstring ends.
    """
    co = compile(open(filepath, encoding="utf-8").read(), filepath, "exec")
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0]
    else:
        print("Could not get the docstring from " + filepath)
        docstring = ""
    return docstring, co.co_firstlineno
