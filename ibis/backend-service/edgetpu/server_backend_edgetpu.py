#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 9/4/20
# Description: 
# ========================================================

import grpc
import backend_service_edgetpu_pb2
import backend_service_edgetpu_pb2_grpc
import logging
import numpy as np
import timeit
import os

from concurrent import futures
from typing import List, Tuple
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.basic import edgetpu_utils, basic_engine

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)

MODEL_HOME = "/models"
PORT = "8080"


def warm_up(engine: basic_engine.BasicEngine, workload_type: str):
    """ Run the engine with some random inputs, return the median of response time"""
    input_size = engine.required_input_array_size()

    res = []
    for i in range(10):
        image = np.random.randint(0, 256, input_size, dtype=np.uint8)

        start_t = timeit.default_timer()
        if workload_type == "detection":
            engine.detect_with_input_tensor(image)
        else:
            engine.classify_with_input_tensor(image)

        end_t = timeit.default_timer()
        res.append((end_t - start_t)*1000)

    return np.median(res)


class InferenceServer(backend_service_edgetpu_pb2_grpc.BackEndServiceServicer):

    def __init__(self, models: List[Tuple], device_path: str=None):
        """
        Load models to specific device
        Args:
            models: A list of tuple, in form (model_filename, workload_type)
            device_path: String represent the device path
        """
        self.logger = logging.getLogger("Inference Server")
        self.logger.setLevel(logging.INFO)

        # Set device path
        if device_path:
            self.device_path = device_path
        else:
            self.device_path = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)[0]

        self.logger.info("Using the device %s" % self.device_path)

        self.models = {}
        for model_filename, workload_type in models:
            model_name = os.path.splitext(model_filename)[0]
            model_path = os.path.join(MODEL_HOME, model_filename)

            if workload_type == 'detection':
                self.models[model_name] = DetectionEngine(model_path, device_path=self.device_path)
                self.logger.info("Successfully loaded detection model %s" % model_name)
            else:
                self.models[model_name] = ClassificationEngine(model_path, device_path=self.device_path)
                self.logger.info("Successfully loaded classification model %s" % model_name)

            inference_time = warm_up(self.models[model_name], workload_type)
            self.logger.info("Finish warm up %s, median response time: %.2f" % (model_name, inference_time))

    def infer(self, request: backend_service_edgetpu_pb2.TPURequest, context):
        """ Handle request """
        start_time = timeit.default_timer()

        image = np.frombuffer(request.image, dtype=np.uint8)
        model_name = request.model_name

        output = []
        if request.type == "detection":
            detection_candidates = self.models[model_name].detect_with_input_tensor(image)
            for candidate in detection_candidates:
                output.append(candidate.bounding_box.tobytes())
                output.append(np.array([candidate.label_id, candidate.score], dtype=np.float).tobytes())
        else:
            classification_candidates = self.models[model_name].classify_with_input_tensor(image)
            for candidate in classification_candidates:
                output.append(np.array(candidate, dtype=np.float).tobytes())

        end_time = timeit.default_timer()
        t_delta = (end_time - start_time) * 1000

        self.logger.info("Finish process %s request in %.2fms" % (model_name, t_delta))

        return backend_service_edgetpu_pb2.TPUResponse(output=output, start_time=start_time,
                                                       end_time=end_time, tpu_time=t_delta)


def main():
    # Environment variable $MODELS should contains model file names separated by comma
    model_filenames = os.environ["MODELS"].split(',')

    # Environment variable $MODEL_TYPES should contains workload types
    # Now only support 'detection' and 'classification'
    model_types = os.environ["MODEL_TYPES"].split(',')

    assert len(model_filenames) == len(model_types), \
        "[ERROR] Got $d models but $d types" % (len(model_filenames), len(model_types))

    models = [(f, t) for f, t in zip(model_filenames, model_types)]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    backend_service_edgetpu_pb2_grpc.add_BackEndServiceServicer_to_server(InferenceServer(models), server)
    server.add_insecure_port('[::]:' + PORT)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
