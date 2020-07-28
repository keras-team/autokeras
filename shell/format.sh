isort --sl autokeras tests
autopep8 -r -i autokeras tests

for i in $(find autokeras tests -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat copyright.txt $i >$i.new && mv $i.new $i
  fi
done
