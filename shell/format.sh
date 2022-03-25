isort --sl --profile=black .
black --line-length 85 .

for i in $(find autokeras benchmark -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 .
