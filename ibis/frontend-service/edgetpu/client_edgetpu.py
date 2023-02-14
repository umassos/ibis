#!/usr/bin/env python

import sys
import logging
import grpc

import frontend_service_edgetpu_pb2
import frontend_service_edgetpu_pb2_grpc

import io
import numpy as np
import timeit

from PIL import Image

# def load_images():
#     images = []
#     for jpgfile in glob.glob('./imgs/*.jpg'):
#         with open(jpgfile, 'rb') as f:
#             images.append([f.read(), jpgfile])
#     return images

def load_images():
    images = []
    for i in range(10):
        buffer = io.BytesIO()
        img = Image.fromarray(np.random.randint(0, 256, (480, 480, 3), dtype=np.uint8))
        img.save(buffer, format='png')

        images.append([buffer.getvalue(), str(i)])
    return images


def run():
    ip_addr = sys.argv[1]

    results = []

    with grpc.insecure_channel('%s:8080' % ip_addr) as channel:
        stub = frontend_service_edgetpu_pb2_grpc.FrontEndServiceStub(channel)

        images = load_images()

        for i in range(3):
            image = images[np.random.randint(len(images))]
            t = np.random.randint(50)
            request = frontend_service_edgetpu_pb2.InferenceRequest(image_name=image[1], image_type=str(t),
                                                                    image=image[0])
            s = timeit.default_timer()
            response = stub.infer(request)
            e = timeit.default_timer()
            output = []

            for i in range(len(response.output)):
                output.append(np.frombuffer(response.output[i], np.float))
            # TotalTime, PreProcess, GPU End to End, GPU Total ,GPU Processing
            result = {"filename": image[1],
                      "output": [ot.shape for ot in output],
                      "total_response_time": (e - s) * 1000, "preprocess_time": response.pre_process_time * 1000,
                      "backend_response_time": response.tpu_total_time * 1000,
                      "backend_service_time": (response.tpu_end_time - response.tpu_start_time) * 1000,
                      "TPU_execution_time": response.tpu_process_time}
            results.append(result)

        for r in results:
            print(r)


if __name__ == '__main__':
    logging.basicConfig()
    run()
