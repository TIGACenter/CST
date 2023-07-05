ARG BASE_CONTAINER=tensorflow/tensorflow:2.8.0-gpu
FROM $BASE_CONTAINER
LABEL maintainer="Felipe Miranda <felipe.miranda@stcmed.com>"

# Added to fix pubkey problem with cuda (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN mkdir main_dir
RUN cd main_dir
WORKDIR /main_dir
RUN mkdir env_files

ADD requirements.txt /main_dir/env_files

RUN pip install --upgrade pip  # may need to delete
RUN pip install -r /main_dir/env_files/requirements.txt

RUN apt-get update --fix-missing
RUN apt-get install -y python3-openslide
RUN apt install graphviz -y

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev ffmpeg

RUN mkdir -p /.local/share/jupyter
RUN chown -R 1000:1000 /.local/share/jupyter
