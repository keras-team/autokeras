FROM python:3.7

RUN pip install flake8==3.7.9 autopep8==1.5
RUN pip install isort

# hack to make isort understand that autokeras is
# a first party module.
RUN git clone --depth=1 https://github.com/keras-team/autokeras.git
RUN pip install -e ./autokeras

WORKDIR /autokeras

CMD ["python", "docker/pre_commit.py"]
