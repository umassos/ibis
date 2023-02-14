import grpc
import backend_service_pb2
import backend_service_pb2_grpc

from concurrent import futures
import logging

import numpy as np

import multiprocessing
import inference_engine as inf
import pycuda.driver as cuda
import timeit
import os
import utils
import time

class InferenceServer(backend_service_pb2_grpc.BackEndServiceServicer):

    def __init__(self, input_file, input_shape):
        self.input_shape = input_shape
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()
        self.model = inf.TRTInferenceEngine("/models/", input_file, input_shape)
        for i in range(3):
            image = np.random.randn(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
            request = inf.Request("Model", 0,0,0,timeit.default_timer(), image)
            self.model.infer(request)
        print("Warming Done")

        self.n = 0

    def InfereImage(self, client_request, context):
        #self.ctx.push()
        self.n = self.n+1
        #print("request ", self.n ,"is in")
        start_time = timeit.default_timer()
        image = np.frombuffer(client_request.image).reshape(self.input_shape)
        #request = inf.Request("effs", 0,0,0,start_time, image)
        #print("Processing ", self.n, "Start")
        request = inf.Request("effs", 0,0,0, 0, image)
        self.ctx.push()
        r = self.model.infer(request,False)
        self.ctx.pop()
        #print("Processing ", self.n, "End")
        output_ids = list(client_request.output_binding)
        #print(r.values())
        output = r.values()
        output = [out.tobytes() for out in output]
        end_time = timeit.default_timer()
        #print(client_request.image_name," ",output_ids, "in", time*1000)
        print("request ", self.n ,"is out")
        #self.ctx.pop()
        return backend_service_pb2.GPUResponse(output_id=output_ids,output=output,start_time = start_time,end_time = end_time, gpu_time = request.response_time())

    def __del__(self):
        self.ctx.pop()


def serve(input_file, input_shape, port, max_wait):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    backend_service_pb2_grpc.add_BackEndServiceServicer_to_server(InferenceServer(input_file, input_shape),server)
    server.add_insecure_port('[::]:'+port)
    server.start()
    server.wait_for_termination()
    engine_process.join()


def load_config():
    input_file = os.environ['fid']        
    shape = os.environ['INPUT_SHAPE'].split('x')
    shape = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
    max_wait = int(os.environ['WAIT_TIME'])
    return input_file, shape, max_wait


def main():
    input_file, input_shape, max_wait = load_config()
    #print("Download ", input_file, " Start")
    #utils.download_file(drive_id, "/models/"+input_file)
    #print("Download Complete")
    serve(input_file, input_shape, "8080", max_wait)


if __name__ == '__main__':
    logging.basicConfig()
    main()
