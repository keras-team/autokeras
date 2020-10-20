from subprocess import CalledProcessError
from subprocess import check_call


def check_bash_call(string):
    check_call(["bash", "-c", string])


def _run_format_and_flake8():
    files_changed = False

    try:
        check_bash_call("sh shell/lint.sh")
    except CalledProcessError:
        check_bash_call("sh shell/format.sh")
        files_changed = True

    if files_changed:
        print("Some files have changed.")
        print("Please do git add and git commit again")
    else:
        print("No formatting needed.")

    if files_changed:
        exit(1)


def run_format_and_flake8():
    try:
        _run_format_and_flake8()
    except CalledProcessError as error:
        print("Pre-commit returned exit code", error.returncode)
        exit(error.returncode)


if __name__ == "__main__":
    run_format_and_flake8()
