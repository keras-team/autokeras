#!/usr/bin/env bash
cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/contribute.md
mkdocs build -c -d ../docs/
echo "autokeras.com" > ../docs/CNAME
cp -r ../docs $1
rm docs/index.md
rm docs/contribute.md
rm -r ../docs/*
