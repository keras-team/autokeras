#!/usr/bin/env bash
cd docs
python autogen.py
mkdocs build
cd ..
echo "autokeras.com" > docs/site/CNAME
git checkout -b gh-pages-temp
git add -f docs/site
git commit -m "gh-pages update"
git subtree split --prefix docs/site -b gh-pages
git push -f origin gh-pages:gh-pages
git branch -D gh-pages
git checkout master
git branch -D gh-pages-temp
