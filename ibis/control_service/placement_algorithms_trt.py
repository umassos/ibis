#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/4/20
# Description: Algorithms for model fusion and queuing
# ========================================================

import os
import re
import random
import numpy as np
import logging
import tensorrt as trt
import subprocess

from typing import List

SYSTEM_THRESHOLD = 0.8
FRONTEND_THRESHOLD = 0.8
MEMORY_THRESHOLD = 0.8
PREPROCESS_SERVICE_RATE = 1000. / 24.
INFEASIBLE_NODES = {}

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(name)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger("Nano placement")
LOGGER.setLevel(logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_mg1_ps_response_time(service_time: List[float], input_rate: List[float]):
    """
    Computing the response time of M/G/1/PS queuing system, using the formula:

        E[R] = 1 / (mu - lambda)

    where mu is the service rate of the system and lambda is the total input rate
    Args:
        service_time: service time in second, for each of loaded models
        input_rate: input rate in queries / seconds, for each of loaded models

    Returns:
        response_time: response time in ms, for each models
        rho: utilization for each models

    """
    assert len(service_time) == len(input_rate), "Length of service time and input rate list have bo be the same."
    total_input_rate = np.sum(input_rate)

    s_system = 0

    for s_i, lamb_i in zip(service_time, input_rate):
        s_system += s_i * lamb_i

    s_system = s_system / total_input_rate
    mu = 1 / s_system

    expected_queuing_delay = 1/(mu-total_input_rate)-(1/mu)
    response_time = [(expected_queuing_delay + s)*1000 for s in service_time]
    rho = [lamb*s for lamb, s in zip(service_time, input_rate)]

    # Debug
    # print("Service time: {}\nInput rate: {}\n Expected response time:{}\n rho: {}"
    #       .format(service_time, input_rate, response_time, rho))

    return response_time, rho


def get_mgc_ps_response_time(lamb: float, mu: float=PREPROCESS_SERVICE_RATE, c=1):
    """
    Return the predicted response time of M/G/c/PS queuing model, given by the equation:

        E[R] = c/lambda * rho / (1-rho)

    Where rho = lambda / mu
    Args:
        lamb: input rate in queries per second
        mu: service rate in queries per second
        c: number of servers

    Returns:
        response_time: predicted response time
    """
    rho = lamb / mu
    response_time = c / lamb * (rho / (1 - rho)) * 1000

    return rho, response_time


def is_memory_feasible(frontend_service, backend_service, node) -> bool:
    """
    Check if the node has enough memory to accommodate the frotnend and backend containers
    Args:
        frontend_service: frontend service
        backend_service:  backend service
        node: target node

    Returns:

    """
    # Clear Memory of backend_service in case of MAAS
    mem = backend_service.memory
    
    if backend_service.model_type == "MAAS" and contains_model(node, backend_service) != -1:
        backend_service.memory = 0

    available_memory = (node.total_memory*MEMORY_THRESHOLD - node.get_service_memory_used())
    required_memory = frontend_service.memory + backend_service.memory
    #print("BE", backend_service.memory)
    #if available_memory < 0 :
    #    print("AV", available_memory)
    #    for service_name, service in node.backend_services.items():
    #        print (service.memory)
    #    for service_name, service in node.frontend_services.items():
    #        print (service.memory)
    #    print(error)

    backend_service.memory = mem

    if available_memory < required_memory:
        LOGGER.info("{}: available memory {:.2f}MB is under the required bar {:.2f}MB."
                    .format(node.node_name, available_memory, required_memory))
        return False
    else:
        return True


def get_gpu_backend_services(node):
    """ Return a dict of GPU backend services """
    return {name: service for name, service in node.backend_services.items() if service.dev_type == 'gpu'}


def is_node_feasible_MaaS(frontend_service, backend_service, node) -> bool:
    """
    Check if both frontend and backend services can run on the specific node,
    Args:
        frontend_service: frontend service
        backend_service: backend service
        node: target node

    Returns:

    """
    if node.node_name in INFEASIBLE_NODES:
        return False
    index = contains_model(node, backend_service)
    if index == -1:
        return is_node_feasible(frontend_service, backend_service, node)

    # Check memory
    if not is_memory_feasible(frontend_service, backend_service, node):
        return False

    # Retrieve service time, input rate and SLA of existing models and
    # add that of incoming model to the list
    service_time = []
    input_rate = []
    sla = []

    for _, service in get_gpu_backend_services(node).items():
        service_time.append(1/service.service_rate)

        if backend_service.model_name == service.model_name:
            input_rate.append(service.input_rate + backend_service.input_rate)
            sla.append(min(service.sla,  backend_service.sla))
        else:
            input_rate.append(service.input_rate)
            sla.append(service.sla)

    # Compute the expected backend process time
    expected_backend_process_time, rho = get_mg1_ps_response_time(service_time, input_rate)

    # If the utilization is higher than system threshold, return False
    system_rho = np.sum(rho)
    if system_rho > SYSTEM_THRESHOLD:
        LOGGER.info("System utilization %.2f is higher than the threshold %.2f, we don't want to overload the system"
                    % (system_rho, SYSTEM_THRESHOLD))
        return False

    # Compute expected pre-process time
    frontend_rho,preprocess_time = get_mgc_ps_response_time(frontend_service.input_rate)
    if frontend_rho > SYSTEM_THRESHOLD:
        LOGGER.info("Front End utilization %.2f is higher than the threshold %.2f, we don't want to overload the system"
                    % (frontend_rho, SYSTEM_THRESHOLD))
        return False
    frontend_process_time = preprocess_time + 10  # 10ms network latency

    expected_frontend_service_rate = [1000/(bs + frontend_process_time) for bs in expected_backend_process_time]
    expected_response_time = [bs + frontend_process_time for bs in expected_backend_process_time]

    is_feasible_load = all([lam < mu*FRONTEND_THRESHOLD for mu, lam in zip(expected_frontend_service_rate, input_rate)])
    is_feasible_sla = all([response_time < s for response_time, s in zip(expected_response_time, sla)])

    if not is_feasible_load or not is_feasible_sla:
        LOGGER.info("{} is NOT feasible, load feasible: {}\tSLA feasible: {}".format(
            node.node_name, is_feasible_load, is_feasible_sla))

        return False
    else:
        LOGGER.info("{} is feasible for {} with input rate {:.2f}".format(
            node.node_name, frontend_service.service_name, backend_service.input_rate))

        return True


def utilization_diff_gpu(node, new_service):
    """ Compute the utilization different before and after placing a service to a node """
    service_time = []
    input_rate = []

    for _, service in get_gpu_backend_services(node).items():
        service_time.append(1/service.service_rate)
        input_rate.append(service.input_rate)

    if not service_time:
        before_util = 0
    else:
        before_util = sum(get_mg1_ps_response_time(service_time, input_rate)[1])

    service_time.append(1 / new_service.service_rate)
    input_rate.append(new_service.input_rate)
    after_util = sum(get_mg1_ps_response_time(service_time, input_rate)[1])

    return after_util - before_util


def is_node_feasible(frontend_service, backend_service, node) -> bool:
    """
    Check if both frontend and backend services can run on the specific node,
    Args:
        frontend_service: frontend service
        backend_service: backend service
        node: target node

    Returns:

    """
    if node.node_name in INFEASIBLE_NODES:
        return False
    
    # Check memory
    if not is_memory_feasible(frontend_service, backend_service, node):
        return False

    # Retrieve service time, input rate and SLA of existing models and
    # add that of incoming model to the list
    service_time = []
    input_rate = []
    sla = []

    for _, service in get_gpu_backend_services(node).items():
        service_time.append(1/service.service_rate)
        input_rate.append(service.input_rate)
        sla.append(service.sla)

    service_time.append(1/backend_service.service_rate)
    input_rate.append(backend_service.input_rate)
    sla.append(backend_service.sla)

    # Compute the expected backend process time
    expected_backend_process_time, rho = get_mg1_ps_response_time(service_time, input_rate)

    # If the utilization is higher than system threshold, return False
    system_rho = np.sum(rho)
    if system_rho > SYSTEM_THRESHOLD:
        LOGGER.info("System utilization %.2f is higher than the threshold %.2f, we don't want to overload the system"
                    % (system_rho, SYSTEM_THRESHOLD))
        return False

    # Compute expected pre-process time
    frontend_rho, preprocess_time = get_mgc_ps_response_time(frontend_service.input_rate)
    if frontend_rho > SYSTEM_THRESHOLD:
        LOGGER.info("Front End utilization %.2f is higher than the threshold %.2f, we don't want to overload the system"
                    % (frontend_rho, SYSTEM_THRESHOLD))
        return False
    frontend_process_time = preprocess_time + 10  # 10ms network latency

    expected_frontend_service_rate = [1000/(bs + frontend_process_time) for bs in expected_backend_process_time]
    expected_response_time = [bs + frontend_process_time for bs in expected_backend_process_time]

    is_feasible_load = all([lam < mu*FRONTEND_THRESHOLD for mu, lam in zip(expected_frontend_service_rate, input_rate)])
    is_feasible_sla = all([response_time < s for response_time, s in zip(expected_response_time, sla)])

    if not is_feasible_load or not is_feasible_sla:
        LOGGER.info("{} is NOT feasible, load feasible: {}\tSLA feasible: {}".format(
            node.node_name, is_feasible_load, is_feasible_sla))

        return False
    else:
        LOGGER.info("{} is feasible for {} with input rate {:.2f}".format(
            node.node_name, frontend_service.service_name, backend_service.input_rate))

        return True


def choose_node_depth_first(feasible_nodes: List, backend_service):
    """
    Choose a feasible node based on depth-first policy:

        1. Sort the feasible node list
        2. Pick the first feasible node
    Args:
        feasible_nodes: A list of feasible nodes

    Returns:
        node: chosen node
    """

    if not feasible_nodes:
        return None
    # we should check that the model is there in all cases
    for node in feasible_nodes:
        if contains_model(node, backend_service) != -1:
            LOGGER.info("Choosing node {} using depth first policy that has the model".format(node.node_name))
            return node
        
    # Sort nodes based on names
    node_names = [n.node_name for n in feasible_nodes]
    index = np.argsort(node_names)[0]

    # Pick first node
    node = feasible_nodes[index]

    LOGGER.info("Choosing node {} using depth first policy".format(node.node_name))
    return node


def choose_node_breadth_first(feasible_nodes: List, backend_service):
    """
    Choose a feasible node based on breadth-first policy:

        1. Sort the feasible node list
        2. Pick the first feasible node
    Args:
        feasible_nodes: A list of feasible nodes

    Returns:
        node: chosen node
    """

    if not feasible_nodes:
        return None

    for node in feasible_nodes:
        if contains_model(node, backend_service) != -1:
            LOGGER.info("Choosing node {} using breadth first policy that has the model".format(node.node_name))
            return node
    
    # Otherwise, place it on a idle node
    for node in feasible_nodes:
        if not get_gpu_backend_services(node):
            LOGGER.info("Choosing node {} for idle node".format(node.node_name))
            return node

    # select
    node_util = []
    for node in feasible_nodes:
        service_time = []
        input_rate = []

        for _, service in get_gpu_backend_services(node).items():
            service_time.append(1/service.service_rate)
            input_rate.append(service.input_rate)
        _, rho = get_mg1_ps_response_time(service_time, input_rate)

        node_util.append(np.sum(rho))

    sort_index = np.argsort(node_util)[0]
    LOGGER.info("Choosing node {} for least loaded node".format(feasible_nodes[sort_index].node_name))
    return feasible_nodes[sort_index]


def choose_node_memory(feasible_nodes: List, backend_service, method):
    """
    Choose a feasible node based on depth-first policy:

        1. Sort the feasible node list
        2. Pick the first feasible node
    Args:
        feasible_nodes: A list of feasible nodes

    Returns:
        node: chosen node
    """

    if not feasible_nodes:
        return None

    # we should check that the model is there in all cases
    for node in feasible_nodes:
        if contains_model(node, backend_service) != -1:
            LOGGER.info("Choosing node {} using memory policy that has the model".format(node.node_name))
            return node
    
    free = np.array([node.total_memory*MEMORY_THRESHOLD - (node.get_service_memory_used() \
                + backend_service.memory) for node in feasible_nodes])
    
    if method == "memory_first": 
        index = 0 # Similar to DFS
    elif method == "memory_best":
        index = np.argmin(free)
    elif method == "memory_worst":
        index = np.argmax(free)

    node = feasible_nodes[index]

    LOGGER.info("Choosing node {} using {} policy".format(node.node_name, method))
    return node


def build_engine(model_path: str, dynamic_shapes: tuple=None, workspace: int=1024) -> str:
    """
    Build TRT engine
    Args:
        model_path: path to the model
        dynamic_shapes: 3-tuple specify (min_shape, max_shape, opt_shape)
        workspace: workspace limit in Mb

    Returns:
        engine_path: path to built engine

    """

    # The engine file will be saved in the same directory
    engine_dir = os.path.dirname(model_path)

    model_name = os.path.basename(model_path)
    model_name = os.path.splitext(model_name)[0]   # model name without suffix
    engine_name = model_name + ".engine"
    engine_path = os.path.join(engine_dir, engine_name)

    if os.path.exists(engine_path):
        print("[WARNING] Engine %s already exists, stop building..." % engine_path)
    else:
        # Call trtexec to build the engine
        build_cmd = ["trtexec",
                     "--onnx=%s" % model_path,
                     "--workspace=%d" % workspace,
                     "--saveEngine=%s" % engine_path,
                     "--buildOnly", "--fp16"]

        if dynamic_shapes:
            build_cmd.append("--explicitBatch")
            build_cmd.append("--minShapes=%s" % dynamic_shapes[0])
            build_cmd.append("--maxShapes=%s" % dynamic_shapes[1])
            build_cmd.append("--optShapes=%s" % dynamic_shapes[2])

        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        assert ret.returncode == 0, "[ERROR] Building engine fail with error: \n %s" % ret.stderr

    return engine_path


def get_trt_engine_memory_usage(engine_path, max_batch_size):
    """
    Profile a TRT engine and return the amount of memory used by this engine
    Args:
        engine_path: path to the engine
        max_batch_size: max batch size

    Returns:
        memory int: memory usage in Mb

    """
    file_size = os.path.getsize(engine_path) / (1 << 20)

    # Load engine
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    device_memory_size = engine.device_memory_size / (1 << 20)

    io_buffer_size = 0
    for binding in engine:
        input_shape = engine.get_binding_shape(binding)
        input_shape[0] = max_batch_size

        print("Shape for %s: " % binding, input_shape)

        mem_size = trt.volume(input_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        io_buffer_size += 2 * (np.dtype(dtype).itemsize * mem_size)

        if engine.binding_is_input(binding):
            print("Input %s total memory usage: %d" % (binding, io_buffer_size))
        else:
            print("Output %s total memory usage: %d" % (binding, io_buffer_size))

    io_buffer_size = io_buffer_size / (1 << 20)

    memory = np.ceil(file_size + device_memory_size + io_buffer_size)

    engine.__del__()

    return memory


def get_trt_service_time(engine_path: str, input_shape: str=""):
    """
    Profile the model to get the service time
    Args:
        engine_path: path to the trt engine
        input_shape: shape in binding:shape format

    Returns:
        service_time: service time in ms

    """
    cmd = ["trtexec", "--loadEngine=%s" % engine_path, "--warmUp=1000"]

    if input_shape:
        cmd.append("--shapes=%s" % input_shape)

    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    assert ret.returncode == 0, "[ERROR] error when profiling the service time of engine %s, error info:" %\
                                (engine_path, ret.stderr)

    match = re.findall("mean: (\d+.\d+)", str(ret.stdout))
    service_time = float(match[0])

    return service_time


def contains_model(node, backend_service):
    i = 0
    for service_name, service in get_gpu_backend_services(node).items():
        if service.model_name == backend_service.model_name and service.model_type == "MAAS":
            return i
        i +=1

    return -1


def get_needed_service(node, backend_service):
    for service_name, service in get_gpu_backend_services(node).items():
        if service.model_name == backend_service.model_name and service.model_type == "MAAS":
            return service

