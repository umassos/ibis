import grpc
import frontend_service_pb2
import frontend_service_pb2_grpc

import backend_service_pb2_grpc
import backend_service_pb2

from concurrent import futures
import logging

import io
import PIL.Image as Image
import numpy as np
import zlib


import sys
import timeit
import os


class FrontEndServer(frontend_service_pb2_grpc.FrontEndServiceServicer):

    def __init__(self, input_shape, gpu_address, output_binding):
        self.input_shape = input_shape
        self.channel = grpc.insecure_channel(gpu_address)
        self.stub = backend_service_pb2_grpc.BackEndServiceStub(self.channel)
        self.output_binding = output_binding
        self.n = 0

    def InfereImage(self, client_request, context):
        self.n = self.n+1
        #print("req start")
        start_time = timeit.default_timer()
        image = Image.open(io.BytesIO(client_request.image))
        image = image.resize((self.input_shape[2], self.input_shape[3]))
        image = np.asarray(image)
        image = image/255
        image = np.transpose(image, (2, 0, 1)).reshape(self.input_shape)
        process_time = timeit.default_timer() - start_time

        # Send Request To Back End
        gpu_start = timeit.default_timer()
        #print("req send to GPU")
        request = backend_service_pb2.GPURequest(
            image_name=client_request.image_name, image_type=client_request.image_type, image=image.tobytes(),
            output_binding=self.output_binding)
        response = self.stub.InfereImage(request)
        gpu_total_time = timeit.default_timer() - gpu_start
        print("Request ",self.n, "Done")
        #print(response.output_id, len(response.output))
        output = []
        for out in response.output:
            output.append(zlib.compress(out))

        return frontend_service_pb2.InferenceResponse(
                output_id=response.output_id,
                output = output,
                front_start_time = start_time,
                pre_process_time = process_time,
                gpu_start_time = response.start_time,
                gpu_process_time = response.gpu_time,
                gpu_end_time = response.end_time,
                gpu_total_time=gpu_total_time, 
                front_end_time = timeit.default_timer())


def serve(input_shape, service_port, gpu_port, output_binding):
    print("Start Service:")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    frontend_service_pb2_grpc.add_FrontEndServiceServicer_to_server(
        FrontEndServer(input_shape, gpu_port,output_binding), server)
    server.add_insecure_port('[::]:'+service_port)
    server.start()
    print("Serive Ready")
    server.wait_for_termination()


def load_config():
    be_host = os.environ['BACKEND_SERVICE_IP']
    be_port = os.getenv('BACKEND_SERVICE_PORT', 8080)
    shape = os.environ['INPUT_SHAPE'].split('x')
    shape = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    output_binding = os.getenv('OUTPUT_BINDING').split(',')
    return be_host, be_port, shape, output_binding


def main():
    be_host, be_port, input_shape,output_binding = load_config()
    serve(input_shape, "8080", be_host+":"+be_port,output_binding)


if __name__ == '__main__':
    logging.basicConfig()
    main()
