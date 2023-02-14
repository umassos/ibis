import sys
import logging
import grpc

import frontend_service_pb2
import frontend_service_pb2_grpc

import pandas as pd
import numpy as np
import timeit
import glob
import zlib
import psutil
from concurrent.futures import ThreadPoolExecutor
import time

def load_requests():
    requests = []
    for jpgfile in glob.glob('./imgs/*.jpg'):
        with open(jpgfile, 'rb') as f:
            name = jpgfile
            image = f.read()
            request = frontend_service_pb2.InferenceRequest(image_name=name, image_type="0", image=image)
            requests.append([name, image, request])
    return requests

def cpu_request(stub, request, results, input_rate, service_rate, rho, util):
    submit_time = timeit.default_timer()
    response = stub.InfereImage(request)
    end_time = timeit.default_timer()
    result = [request.image_name, submit_time, end_time,response.front_start_time, response.front_end_time, input_rate, service_rate, rho, util]
    results.append(result)

def run(ip_addr, requests, service_rate, rho, duration):
    pool = ThreadPoolExecutor(max_workers=5000)
    requests_count = len(requests)
    util = psutil.cpu_percent(interval=None, percpu=True)[3]
    with grpc.insecure_channel('%s:8080' % ip_addr) as channel:
        stub = frontend_service_pb2_grpc.FrontEndServiceStub(channel)

        # Warmup
        for i in range(10):
            name, image,request = requests[np.random.randint(requests_count)]
            _ = stub.InfereImage(request)
        
        input_rate = rho * service_rate
        iterations = int(duration * input_rate)
        util = psutil.cpu_percent(interval=None, percpu=True)[3]
        for i in range(iterations):
            if i %10 == 0:
                util = psutil.cpu_percent(interval=None, percpu=True)[3]
            name, image,request = requests[np.random.randint(requests_count)]
            ret = pool.submit(cpu_request, stub, request, results, input_rate, service_rate, rho, util)
            time.sleep(np.random.exponential(1 / input_rate))
        
        ret.result()
        pool.shutdown(wait=True)
        return results

if __name__ == '__main__':
    logging.basicConfig()
    ip_addr = sys.argv[1]
    requests = load_requests()
    rhos = [.1, .2, .3, .4, .5, .6,.7, .8]
    service_rate = 170
    duration = 100
    results = []

    for rho in rhos:
        print("Run with Rho = ", rho)
        result = run(ip_addr, requests, service_rate, rho, duration)
        results.extend(result)

    df = pd.DataFrame(results, columns=["Image","submit_time","end_time","front_start_time","front_end_time", "input_rate", "service_rate", "rho", "util"])
    df.to_csv("results/queuing_cpu_withutil_10xworker.csv", index=False)
