# node shell/generate_json.js
gh api -H "Accept: application/vnd.github+json" /repos/keras-team/autokeras/contributors --paginate > response.json
sed "s/\]\[/,/g" response.json > contributors.json
rm response.json
mkdir avatars
python shell/contributors.py avatars > docs/templates/img/contributors.svg
rm contributors.json
rm -rf avatars
