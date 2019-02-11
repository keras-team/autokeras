
for uf in unittest_*.py
do
  echo Testing $uf
  python $uf
  rm -rf *.py[co]
done
