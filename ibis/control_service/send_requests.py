#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 9/25/20
# Description: 
# ========================================================

import io
import os
import time
import timeit
import logging
import grpc
import yaml
import csv
import numpy as np
import argparse
import threading
import multiprocessing

import frontend_service_edgetpu_pb2
import frontend_service_edgetpu_pb2_grpc

from PIL import Image
from typing import List
from kubernetes import client, config
from control_service_base import num_generator
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_images():
    """ Pre-load a set of images as PNG format """
    images = []
    for i in range(100):
        buffer = io.BytesIO()
        img = Image.fromarray(np.random.randint(0, 256, (480, 480, 3), dtype=np.uint8))
        img.save(buffer, format='png')

        images.append([buffer.getvalue(), str(i)])

    return images


def get_service_by_name(service_name: str, services: List):
    """ Retrive the service by service name """
    for service in services:
        if service.metadata.name == service_name:
            return service

    msg = "Service {} not found. Available services: {}".format(service_name,
                                                                [service.metadata.name for service in services])
    raise ValueError(msg)


def send_request(request, stub, service_name, node_name, input_rate, result_collector: List, request_name):
    """ Submit a request and save result to result collector """
    # Debug
    # print("[INFO] Sending %s" % request_name)

    start_t = timeit.default_timer()
    response = stub.infer(request)
    end_t = timeit.default_timer()

    result = {"service_name": service_name,
              "node": node_name,
              "reqest_name": request_name,
              "input_rate": input_rate,
              "start_time": start_t,
              "end_t": end_t,
              "front_end_time": response.front_end_time,
              "front_start_time": response.front_start_time,
              "total_response_time": (end_t - start_t) * 1000,
              "preprocess_time": response.pre_process_time * 1000,
              "backend_response_time": response.tpu_total_time * 1000,
              "backend_service_time": (response.tpu_end_time - response.tpu_start_time) * 1000,
              "TPU_execution_time": response.tpu_process_time}

    result_collector.append(result)
    # print("[INFO] finish %s" % request_name)


def simulate_requests(service_name: str, node_name: str, ip_address: str, input_rate: float,
                      inputs: List, duration: int, result_collector: List):
    """
    Generate requests in a given rate, save result back to result collector
    Args:
        service_name: name
        node_name: which node the server is assigned to
        ip_address: IP address of the frontend server in ip:port format
        input_rate: input rate
        inputs: pre-loaded inputs
        duration: duration of this experiment
        result_collector: list to put results

    Returns:

    """
    iterations = int(np.ceil(input_rate * duration))
    worker_threads = ThreadPoolExecutor(max_workers=20)
    futures = []

    with grpc.insecure_channel(ip_address) as channel:
        stub = frontend_service_edgetpu_pb2_grpc.FrontEndServiceStub(channel)

        start_t = time.time()
        for i in range(iterations):
            image = inputs[np.random.randint(len(inputs))]
            t = np.random.randint(50)
            request = frontend_service_edgetpu_pb2.InferenceRequest(image_name=image[1], image_type=str(t),
                                                                    image=image[0])

            futures.append(worker_threads.submit(send_request, request, stub, service_name, node_name, input_rate,
                                                 result_collector, service_name + "-%d" % i))

            # if worker_threads._work_queue.qsize() > 0:
            #     logger.warning("%s with input rate %.2f has queue size %d" % (service_name, input_rate,
            #                                                                   worker_threads._work_queue.qsize()))
            time.sleep(np.random.exponential(1 / input_rate))

        total_time = time.time() - start_t
        print("[INFO] {} finish in time {:.2f}s, total {} requests, input rate: {:2f}".format(service_name,
                                                                                              total_time,
                                                                                              iterations,
                                                                                              input_rate))
        # Wait until all requests finished
        for f in futures:
            _ = f.result()
        worker_threads.shutdown(wait=True)


def main():
    """ Load benchmark file, generate request workloads """
    parser = argparse.ArgumentParser(description="AI workload requests simulator")
    parser.add_argument("-f", "--workload-file", required=True, dest="workload_filename", type=str,
                        help="Path to the workload file")
    parser.add_argument("-t", "--time", default=60, type=int, dest="time", help="Duration of the experiment")
    args = parser.parse_args()

    # Load benchmark
    logger.info("Reading benchmark file %s" % args.workload_filename)
    with open(args.workload_filename, 'r') as f:
        workload = yaml.load(f.read())


    # Get k8s interface
    config.load_kube_config()
    k8s_core_v1 = client.CoreV1Api()
    k8s_services = k8s_core_v1.list_service_for_all_namespaces().items

    suffix_generator = num_generator()

    manager = multiprocessing.Manager()
    processes = defaultdict(list)
    result_collector = manager.list()
    duration = args.time
    inputs = load_images()

    # worker_pool = ThreadPoolExecutor(max_workers=100)

    for model in workload:
        # re-construct the service name
        # suffix = next(suffix_generator)
        # model_name = os.path.splitext(model["model_filename"])[0]
        # frontend_service_name = model_name.lower() + "-frontend-" + suffix
        # frontend_service_name = frontend_service_name.replace('_', '-')
        frontend_service_name = model["frontend_service_name"]
        input_rate = model["input_rate"]

        # Get ip address
        service = get_service_by_name(frontend_service_name, k8s_services)
        ip_address = service.spec.cluster_ip + ':' + "8080"

        # Get node
        selector = "app=%s" % frontend_service_name
        node = k8s_core_v1.list_pod_for_all_namespaces(label_selector=selector).items[0]
        node_name = node.spec.node_name

        logger.info("Creating client for %s with input rate: %.2f" % (frontend_service_name, input_rate))
        p = multiprocessing.Process(target=simulate_requests, args=(frontend_service_name, node_name, ip_address,
                                                                    input_rate, inputs, duration,
                                                                    result_collector))
        processes[node_name].append(p)

    for node_name, services in processes.items():
        start_t = time.time()
        logger.info("Start sending requests to %s" % node_name)
        for p in services:
            p.start()

        for i in range(int(args.time / 10)):
            logger.info("Time left for this experiment: %.2fs" %
                        (args.time - (time.time() - start_t)))
            time.sleep(10)

        for p in services:
            p.join()

    # worker_pool.shutdown(wait=True)

    workload_basename = "edgetpu_workload"
    output_file = os.path.join("results/", workload_basename + ".csv")

    logger.info("Saving result to %s" % output_file)
    keys = result_collector[0].keys()
    with open(output_file, 'w') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_collector)


if __name__ == '__main__':
    main()
