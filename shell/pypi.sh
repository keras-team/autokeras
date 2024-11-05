#!/usr/bin/env bash
rm dist/*
python -m build
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
