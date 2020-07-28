isort --sl -c autokeras tests
flake8 autokeras tests
for i in $(find autokeras tests -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    exit 1
  fi
done
