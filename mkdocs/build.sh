#!/usr/bin/env bash
sh update.sh
mkdir ../docs
mkdocs build -c -d ../docs/
echo "autokeras.com" > ../docs/CNAME
