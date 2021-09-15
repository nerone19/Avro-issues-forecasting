
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y \
 git \
 curl \
 ca-certificates \
 python3 \
python3-pip \
 sudo \
 && rm -rf /var/lib/apt/lists/*


# make our user own its own home directory




WORKDIR /app

COPY . .
CMD python3 ./src/server.py

