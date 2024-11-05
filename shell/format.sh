#!/bin/bash
set -Euo pipefail

ruff check --fix .

ruff format .

for i in $(find autokeras benchmark -name '*.py')
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done
