FROM nvcr.io/nvidia/tensorrt:20.07-py3

RUN python -m pip install grpcio --ignore-installed

RUN python -m pip install grpcio-tools

RUN python -m pip install pandas minio

RUN mkdir /app

ADD backend_service_pb2.py /app/

ADD backend_service_pb2_grpc.py /app/

ADD server_backend.py /app/

ADD server_backend_single_process.py /app/

ADD inference_engine.py /app/

ADD common.py /app/

ADD utils.py /app/

WORKDIR /app/

ENV WAIT_TIME=5

ENV INPUT_SHAPE=1x3x224x224

ENTRYPOINT python server_backend_single_process.py 

#ENTRYPOINT python server_backend.py
