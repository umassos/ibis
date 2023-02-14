import sys
import logging
import grpc

import frontend_service_pb2
import frontend_service_pb2_grpc

import numpy as np
import timeit
import glob
import zlib

def load_images():
    images = []
    for jpgfile in glob.glob('./imgs/*.jpg'):
        with open(jpgfile, 'rb') as f:
            images.append([f.read(),jpgfile])
    return images


def run():
    ip_addr = sys.argv[1]
    shapes = sys.argv[2]
    shapes = shapes.split(",")
    shapes = [tuple([int(i) for i in shape.split('x')]) for shape in shapes]
    results = []
    with grpc.insecure_channel('%s:8080' % ip_addr) as channel:
        stub = frontend_service_pb2_grpc.FrontEndServiceStub(channel)
        images = load_images()
        
        for i in range(3):
            image = images[np.random.randint(len(images))]
            t = np.random.randint(50)
            request = frontend_service_pb2.InferenceRequest(image_name=image[1], image_type=str(t), image=image[0])
            s = timeit.default_timer()
            response = stub.InfereImage(request)
            e = timeit.default_timer()
            output = []
            for i in range(len(response.output)):
                output.append(np.frombuffer(zlib.decompress(response.output[i]),np.float16).reshape(shapes[i]))
            # TotalTime, PreProcess, GPU End to End, GPU Total ,GPU Processing
            result = {"filename": image[1], 
                      "output_id": response.output_id,
                      "output": [ot.shape for ot in output],
                      "total_response_time": (e-s)*1000, "preprocess_time": response.pre_process_time * 1000,
                      "backend_response_time": response.gpu_total_time * 1000,
                      "backend_service_time": (response.gpu_end_time - response.gpu_start_time)*1000,
                      "GPU_execution_time": response.gpu_process_time * 1000}
            results.append(result)
        
        for r in results:
            print(r)


if __name__ == '__main__':
    logging.basicConfig()
    run()
