#!/usr/bin/env bash
mkdocs build -c -d ../docs/
echo "autokeras.com" > ../docs/CNAME
cp -r ../docs $1
