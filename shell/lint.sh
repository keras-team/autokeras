isort --sl -c autokeras tests
if ! [ $? -eq 0 ]
then
    exit 1
fi
flake8 autokeras tests
if ! [ $? -eq 0 ]
then
    exit 1
fi
black --check --line-length 85 autokeras tests
if ! [ $? -eq 0 ]
then
    exit 1
fi
for i in $(find autokeras tests -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    exit 1
  fi
done
