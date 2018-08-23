#!/usr/bin/env bash
sh update.sh
mkdocs build -c -d ../docs/
echo "autokeras.com" > ../docs/CNAME
cp -r ../docs
