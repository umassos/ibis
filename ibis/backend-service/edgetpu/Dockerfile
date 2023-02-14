FROM lemariva/raspbian-edgetpu

RUN python3 -m pip install grpcio --ignore-installed && \
    python3 -m pip install grpcio-tools && \
    pip3 install protobuf

RUN mkdir /app

COPY backend_service_edgetpu_pb2.py \
     backend_service_edgetpu_pb2_grpc.py \
     server_backend_edgetpu.py /app/


WORKDIR /app/

CMD python3 server_backend_edgetpu.py
