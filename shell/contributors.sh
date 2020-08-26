node shell/generate_json.js
mkdir avatars
python shell/contributors.py avatars > docs/templates/img/contributors.svg
rm contributors.json
rm -rf avatars
