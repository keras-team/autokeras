#!/usr/bin/env bash
cd mkdocs
python autogen.py
mkdocs build
echo "autokeras.com" > ../docs/site/CNAME