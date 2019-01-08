#!/usr/bin/env bash
rm -rf docs/temp
mkdir docs/temp
# cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/temp/contribute.md
cp ../nas/README.md docs/temp/nas.md
python autogen.py
