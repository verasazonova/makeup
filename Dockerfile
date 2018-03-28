FROM python:2

RUN apt-get update -y
RUN apt-get install -y libopencv-dev python-opencv

COPY . /makeup
WORKDIR /makeup

RUN pip install -r requirements.txt
