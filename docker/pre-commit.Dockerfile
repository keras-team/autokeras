FROM python:3.7

RUN pip install flake8==3.7.9 autopep8==1.5
RUN pip install isort==4.3.21

WORKDIR /autokeras
CMD ["python", "docker/pre_commit.py"]
