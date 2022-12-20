# AutoKeras Documentation

The source for AutoKeras documentation is in this directory.
Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- Install dependencies: `pip install -r docs/requirements.txt`
- `pip install -e .` to make sure that Python will import your modified version of AutoKeras.
- From the root directory, `cd` into the `docs/` folder and run:
    - `python autogen.py`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](http://localhost:8000)
    - `mkdocs build`    # Builds a static site in `site/` directory

## Generate contributors list
- Prerequisites:
    - Install Pillow: `pip install Pillow`
- Generate:
    - Run: `sh shell/contributors.sh`
    - The generated file is: `docs/templates/img/contributors.svg`