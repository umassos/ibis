FROM lemariva/raspbian-edgetpu

RUN apt-get update

RUN python3 -m pip install grpcio && \
    python3 -m pip install grpcio-tools && \
    python3 -m pip install Pillow numpy

RUN mkdir /app

COPY backend_service_edgetpu_pb2.py \
     backend_service_edgetpu_pb2_grpc.py \
     frontend_service_edgetpu_pb2.py \
     frontend_service_edgetpu_pb2_grpc.py \
     server_frontend_edgetpu.py /app/

WORKDIR /app/


CMD python3 server_frontend_edgetpu.py