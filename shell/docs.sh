#!/usr/bin/env bash
cd docs
python autogen.py
mkdocs build
echo "autokeras.com" > ../docs/site/CNAME