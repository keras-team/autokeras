FROM python:3.8

RUN pip install ruff

WORKDIR /autokeras
CMD ["python", "docker/pre_commit.py"]
