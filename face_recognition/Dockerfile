FROM ubuntu:18.04
RUN mkdir -p /work
WORKDIR /work
RUN apt-get -y update
RUN apt-get -y install wget
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sh /work/Miniconda3-latest-Linux-x86_64.sh -b
RUN apt-get -y install vim g++
RUN /root/miniconda3/bin/conda create -n env python=3.6.6 pip
RUN echo "source activate env" > ~/.bashrc
ENV PATH /root/miniconda3/bin/:$PATH
RUN apt-get -y update
RUN apt-get -y install jq
RUN apt-get -y install curl
RUN apt-get -y install git
RUN apt-get -y install g++ gfortran
RUN apt-get -y install make
RUN apt-get -y install cmake
RUN /root/miniconda3/envs/env/bin/pip install sh face_recognition flask
ADD server /work/server
WORKDIR /work/server
CMD /work/server/start_server
