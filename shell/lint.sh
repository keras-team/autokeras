isort --sl -c .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
flake8 .
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
black --check --line-length 85 .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
for i in $(find autokeras tests benchmark -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
  fi
done
