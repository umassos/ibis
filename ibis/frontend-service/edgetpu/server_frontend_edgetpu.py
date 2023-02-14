#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 9/4/20
# Description:
# ========================================================

import grpc
import frontend_service_edgetpu_pb2
import frontend_service_edgetpu_pb2_grpc
import backend_service_edgetpu_pb2
import backend_service_edgetpu_pb2_grpc

import logging

import io
import PIL.Image as Image
import numpy as np
import threading

import time
import timeit
import os

from concurrent import futures
from typing import List

PORT = "8080"
THREAD_POOL_SIZE = 30

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(name)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)

WORKER_POOL = futures.ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


# gRPC server instance
class FrontEndServer(frontend_service_edgetpu_pb2_grpc.FrontEndServiceTPUServicer):

    def __init__(self, input_shape: List, model_name: str, workload_type: str, backend_address: str,
                 max_input_rate: float=-1):
        """
        Intialize the server, create stub to backend container
        Args:
            input_shape: input shape
            model_name: model name
            workload_type: workload type, 'detection' | 'classification'
            backend_address: backend address in ip:port format
        """
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.INFO)

        self.input_shape = input_shape
        self.model_name = model_name
        self.workload_type = workload_type
        self.backend_address = backend_address

        # Create gRPC stub to backend
        self.channel = grpc.insecure_channel(self.backend_address)
        self.stub = backend_service_edgetpu_pb2_grpc.BackEndServiceStub(self.channel)

        self.max_input_rate = max_input_rate
        # To record request status
        self.log_requests = []
        self.logger.info("Frontend service for %s started with thread pool size %d" %
                         (self.model_name, THREAD_POOL_SIZE))

        self.refill_thread = None
        self.token_bucket = None
        self.token_bucket_lock = threading.Lock()
        if self.max_input_rate != -1:
            self.logger.info("Maximum input rate {:.2f}".format(self.max_input_rate))
            self.token_bucket = max_input_rate
            self.refill_thread = threading.Thread(target=self._refill_bucket)
            self.refill_thread.start()

    def _refill_bucket(self):
        """ Refill bucket base on max input rate """
        while True:
            with self.token_bucket_lock:
                self.token_bucket = min(self.max_input_rate, self.token_bucket+1.)

            time.sleep(1/self.max_input_rate)

    def infer(self, request: frontend_service_edgetpu_pb2.InferenceRequestTPU, context):
        """
        Receive requests from client, preprocess the request and send it to backend container for inference
        Args:
            request: request from client
            context:

        Returns:

        """
        start_time = timeit.default_timer()
        if self.token_bucket is not None:
            is_accepted = False

            while not is_accepted:
                with self.token_bucket_lock:
                    if self.token_bucket >= 1:
                        self.token_bucket -= 1
                        is_accepted = True

        pre_process_start_time = timeit.default_timer()
        image = Image.open(io.BytesIO(request.image))
        image = image.resize((self.input_shape[1], self.input_shape[2]))        # Assume images are in HWC format
        image = np.asarray(image, dtype=np.uint8)

        process_time = timeit.default_timer() - pre_process_start_time

        # check pool queue
        self.logger.info("Queue: {}\t Threads: {}".format(WORKER_POOL._work_queue.qsize(),
                                                          len(WORKER_POOL._threads)))

        # Send Request To Back End
        tpu_start = timeit.default_timer()
        request = backend_service_edgetpu_pb2.TPURequest(image_name=request.image_name,
                                                         image_type=request.image_type,
                                                         image=image.tobytes(), model_name=self.model_name,
                                                         type=self.workload_type)

        response = self.stub.infer(request)
        tpu_total_time = timeit.default_timer() - tpu_start
        end_time = timeit.default_timer()
        self.log_requests.append({"start_t": start_time, "end_t": end_time})

        return frontend_service_edgetpu_pb2.InferenceResponseTPU(
                output=response.output,
                front_start_time=start_time,
                pre_process_time=process_time,
                tpu_start_time=response.start_time,
                tpu_process_time=response.tpu_time,
                tpu_end_time=response.end_time,
                tpu_total_time=tpu_total_time,
                front_end_time=end_time)


    def get_status(self, request: frontend_service_edgetpu_pb2.GetStatus, context):
        """
        Compute the input rate and mean response time of latest n requests
        Args:
            request: should be empty
            context: grpc context

        Returns:

        """
        # Response time of last n requests
        n = request.num_requests
        log_requests = self.log_requests[-n:]

        # No enough samples
        if len(log_requests) < n:
            response_time = 0
            input_rate = 0
        else:
            response_time = np.mean([(r["end_t"] - r["start_t"])*1000 for r in log_requests])
            input_rate = len(log_requests) / (timeit.default_timer() - min([r["start_t"] for r in log_requests]))

        return frontend_service_edgetpu_pb2.Status(input_rate=input_rate, response_time=response_time)


def main():
    # Load configuration
    backend_host = os.environ['BACKEND_SERVICE_IP']
    backend_port = os.getenv('BACKEND_SERVICE_PORT', "8080")
    backend_address = backend_host + ":" + backend_port

    input_shape = os.environ['INPUT_SHAPE'].split('x')
    input_shape = [int(x) for x in input_shape]
    input_shape[0] = 1      # EdgeTPU can only run with batch size of 1

    # model name should be the model filename without extension
    model_name = os.environ["MODEL_NAME"]

    # workload type should be either 'detection' or 'classification'
    workload_type = os.environ["MODEL_TYPE"]

    # Input rate
    input_rate = float(os.environ["INPUT_RATE"])

    # Start gRPC server

    server = grpc.server(WORKER_POOL)

    frontend = FrontEndServer(input_shape, model_name, workload_type, backend_address, input_rate)
    frontend_service_edgetpu_pb2_grpc.add_FrontEndServiceTPUServicer_to_server(frontend, server)
    server.add_insecure_port("[::]:" + PORT)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
