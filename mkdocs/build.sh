#!/usr/bin/env bash
mkdocs build -c -d ../docs/
echo "autokeras.com" > ../docs/CNAME
