isort --sl autokeras tests benchmark
black --line-length 85 autokeras tests benchmark

for i in $(find autokeras tests benchmark -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 autokeras tests benchmark
