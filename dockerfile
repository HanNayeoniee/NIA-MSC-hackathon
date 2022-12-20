# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
LABEL email="nayeon2.han@gmail.com"
LABEL name="HanNayeoniee"
LABEL version="1.0.0"
LABEL description="ET5-base for NIA MSC Hackathon" 
COPY ./ /workspace/
WORKDIR /workspace
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt
RUN pip install -r requirements.txt
RUN export GIT_PYTHON_REFRESH=quiet
