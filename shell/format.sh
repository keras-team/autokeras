isort --sl autokeras tests
black --line-length 85 autokeras tests

for i in $(find autokeras tests -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 autokeras tests
