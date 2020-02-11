FROM python:3.7

# hack to make isort understand that autokeras is
# a first party module.
RUN git clone --depth=1 https://github.com/keras-team/autokeras.git
RUN pip install --no-deps -e ./autokeras

RUN pip install flake8==3.7.9 autopep8==1.5 isort==4.3.21

WORKDIR /autokeras

CMD ["python", "docker/pre_commit.py"]
