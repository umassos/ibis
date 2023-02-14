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

    def __init__(self, queue, input_shape):
        self.queue = queue
        self.input_shape = input_shape
        if input_shape[0] !=1:
            self.input_shape = (1,input_shape[1], input_shape[2], input_shape[3])
        self.n = 0

    def InfereImage(self, client_request, context):
        self.n = self.n+1
        #print("request ", self.n ,"is in")
        start_time = timeit.default_timer()
        image = np.frombuffer(client_request.image).reshape(self.input_shape)
        #request = inf.Request("effs", 0,0,0,start_time, image)
        engine_conn, thread_conn = multiprocessing.Pipe()
        #print(list(client_request.output_binding), client_request.output_binding)
        self.queue.put((engine_conn, image, list(client_request.output_binding)))
        output_ids, output, time  = thread_conn.recv()
        output = [out.tobytes() for out in output]
        end_time = timeit.default_timer()
        #print(client_request.image_name," ",output_ids, "in", time*1000)
        #print("request ", self.n ,"is out")
        return backend_service_pb2.GPUResponse(output_id=output_ids,output=output,start_time = start_time,end_time = end_time, gpu_time = time)

    def __del__(self):
        self.ctx.pop()

def run_engine_process(input_file, input_shape, queue, max_wait):
    print("Warming Start")
    n = 0
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    model = inf.TRTInferenceEngine("/models/",input_file, input_shape)
    for i in range(3):
        image = np.random.randn(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
        request = inf.Request("Model", 0,0,0,timeit.default_timer(), image)
        model.infer(request)
    print("Warming Done")
    batch = input_shape[0]
    print(batch)
    while 1:
        for i in range(0, max_wait):
            if queue.qsize()>=batch:
                break
            time.sleep(0.001)
        
        requests = []
        for i in range(0,batch):
            if not queue.empty():
                requests.append(queue.get())    
            else:
                break
        if len(requests) == 0:
            continue
        
        #print("Current Batch= ",len(requests))
        fullimage = np.empty(shape=(len(requests),input_shape[1],input_shape[2],input_shape[3]), dtype=float)
        for i in range(0, len(requests)):
            conn, img, output_binding = requests[i]
            fullimage[i,:,:,:] = img

        request = inf.Request("effs", 0, 0, 0, 0,fullimage)
        r = model.infer(request, False)
        #print(r)
        n = n+1
        #print("procesing ", n, "done")
        for i in range(0, len(requests)):
            conn, img, output_binding = requests[i]
            output_ids = []
            output = []
            #print(output_binding)
            for op in output_binding:
                output_ids.append(op)
                #print(r[(op, i)].shape)
                output.append(r[(op, i)])
            conn.send((output_ids, output, request.response_time()))


def serve(input_file, input_shape, port, max_wait):
    engine_conn, thread_conn = multiprocessing.Pipe() 
    queue = multiprocessing.Queue()
    engine_process = multiprocessing.Process(target=run_engine_process, args=(input_file, input_shape, queue, max_wait, )) 
    engine_process.start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    backend_service_pb2_grpc.add_BackEndServiceServicer_to_server(InferenceServer(queue, input_shape),server)
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
