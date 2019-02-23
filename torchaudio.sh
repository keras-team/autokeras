#!/usr/bin/env bash
mkdir external_packages
cd external_packages
git clone https://github.com/pytorch/audio.git
cd audio && python setup.py install
cd ..
cd ..