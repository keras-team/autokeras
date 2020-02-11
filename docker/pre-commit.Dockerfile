FROM python:3.7

RUN pip install flake8==3.7.9 autopep8==1.5
RUN pip install git+https://github.com/timothycrosley/isort.git

WORKDIR /autokeras
CMD ["python", "docker/pre_commit.py"]
