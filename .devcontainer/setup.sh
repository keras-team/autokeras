sudo pip install -e ".[tests]"
echo "sh shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit

sudo apt install wget -y
wget https://github.com/cli/cli/releases/download/v2.17.0/gh_2.17.0_linux_amd64.deb -P /tmp
sudo apt install /tmp/gh_2.17.0_linux_amd64.deb -y
