pip install -e ".[tests]"
echo "sh shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit