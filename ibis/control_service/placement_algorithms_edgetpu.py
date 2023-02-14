#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/4/20
# Description: Algorithms for model fusion and queuing
# ========================================================

import numpy as np
import logging

from typing import List

INFEASIBLE_NODES = {}
SYSTEM_THRESHOLD = 0.8
FRONTEND_THRESHOLD = 08
MEMORY_THRESHOLD = 0.8
PREPROCESS_SERVICE_RATE = 1000. / 20.

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(name)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger("EdgeTPU placement")
LOGGER.setLevel(logging.INFO)


def get_mg1_fcfs_response_time(service_time: List[float], switch_overhead: List[float], input_rate: List[float]):
    """
    Compute the expected resnse time of M/G/1/FCFS queue
    Args:
        service_time: service time in second for each of the models on EdgeTPU without switch overhead
        switch_overhead: switch overhead in ms for each model on EdgeTPU
        input_rate: input rate in frames per seconds

    Returns:
        expected_response_time: List of float representing the expected repsonse time for each models
        rho: The utilization for each models

    """
    assert len(service_time) == len(switch_overhead), "Length of input arrays should be equal"
    assert len(input_rate) == len(switch_overhead), "Length of input arrays should be equal"
    total_input_rate = np.sum(input_rate)

    # Compute the expected tpu process time for each of models using the formula
    # `TPU_process_time = service_time + p(need context switch) * switch_overhead`
    # Where the probability of needing context switch is the probability that the
    # previous request in queue is not the same as the current one. In this case
    # is just 1 - input_rate / total_input_rate
    expected_tpu_response_time = []
    for s, o, lamb in zip(service_time, switch_overhead, input_rate):
        s = s * 1000  # second to ms
        expected_tpu_response_time.append(s + (1 - lamb / total_input_rate) * o)

    # Compute system service time S as the sum of expected tpu response time weighted by input rate
    system_service_time = 0
    system_service_time_second_moment = 0
    for lamb_i, s_i in zip(input_rate, expected_tpu_response_time):
        system_service_time += lamb_i / total_input_rate * s_i
        system_service_time_second_moment += lamb_i / total_input_rate * s_i ** 2

    # utilization of each model
    rho = [lamb_i * s_i / 1000 for lamb_i, s_i in zip(input_rate, expected_tpu_response_time)]
    system_rho = np.sum(rho)

    # Pollaczek–Khinchine formula
    waiting_time = total_input_rate * system_service_time_second_moment / (2 * (1 - system_rho))
    waiting_time = waiting_time / 1e3  # To ms

    expected_response_time = [s + waiting_time for s in expected_tpu_response_time]

    return expected_response_time, rho


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

    return response_time


def get_tpu_backend_service(node):
    """ Get TPU backend service run on node, return None if not found """
    if not node.backend_services:
        return None

    tpu_service = [s for _, s in node.backend_services.items() if s.dev_type == 'tpu']
    if tpu_service:
        return tpu_service[0]
    else:
        return None


def is_memory_feasible(frontend_service, backend_service, node) -> bool:
    """ Check if memory satisfy the requirement """
    available_memory = (node.total_memory * MEMORY_THRESHOLD - node.get_service_memory_used())
    required_memory = frontend_service.memory + backend_service.memory

    # Check if memory satisfy the requirement
    if available_memory < required_memory:
        LOGGER.info("{}: available memory {:.2f}MB is under the required bar {:.2f}MB."
                    .format(node.node_name, available_memory, required_memory))
        return False
    else:
        return True


def is_node_feasible(frontend_service, backend_service, node) -> bool:
    """
    Check if both frontend and backend services can run on the specific node,
    Args:
        frontend_service: New frontend service
        backend_service: New backend service
        node: target node

    Returns:

    """
    if node.node_name in INFEASIBLE_NODES:
        return False

    # Check if memory satisfy the requirement
    if not is_memory_feasible(frontend_service, backend_service, node):
        return False

    exist_backend_service = get_tpu_backend_service(node)
    # If the node is idle
    if not exist_backend_service:

        service_time = [1/backend_service.service_rates[0]]
        switch_overhead = [backend_service.switch_overhead[0]]
        input_rate = [backend_service.input_rates[0]]
        sla = [backend_service.sla[0]]

    else:
        # check if the incoming model have already existed and find its index if exist
        exist_model_index = model_index(backend_service, exist_backend_service)

        # Add new model's service time to existing models
        service_time = [1/mu for mu in exist_backend_service.service_rates]

        if exist_model_index == -1:
            # If the incoming model is not exist on the node, append its service time to the node
            service_time.append(1/backend_service.service_rates[0])

        # Add new model's switch overhead to existing models
        switch_overhead = [overhead for overhead in exist_backend_service.switch_overhead]
        if exist_model_index == -1:
            # If the incoming model is not exist on the node, append its switch overhead to the node
            switch_overhead.append(backend_service.switch_overhead[0])

        # Add new model's input rate to existing models
        input_rate = [lamb for lamb in exist_backend_service.input_rates]
        if exist_model_index == -1:
            # If the incoming model is not exist on the node, append its input rate
            input_rate.append(backend_service.input_rates[0])
        else:
            # If the incoming model has already existed on the node, just increase the existing input rate
            input_rate[exist_model_index] += backend_service.input_rates[0]

        # Add new model's SLA to existing models
        sla = [s for s in exist_backend_service.sla]
        if exist_model_index == -1:
            # If the incoming model is not exist on the node, append a new SLA
            sla.append(backend_service.sla[0])
        else:
            # If the incoming model is exist on the node and we are going to shared it,
            # the SLA would be the minimal of the existing and incoming
            sla[exist_model_index] = min(sla[exist_model_index], backend_service.sla[0])

    # Compute the expected backend process time
    expected_backend_process_time, rho = get_mg1_fcfs_response_time(service_time, switch_overhead, input_rate)

    # If the utilization is higher than system threshold, return False
    system_rho = np.sum(rho)
    if system_rho > SYSTEM_THRESHOLD:
        LOGGER.info("System utilization %.2f is higher than the threshold %.2f, we don't want to overload the system"
                    % (system_rho, SYSTEM_THRESHOLD))
        return False

    # Pre-process
    preprocess_time = get_mgc_ps_response_time(frontend_service.input_rate)
    frontend_process_time = preprocess_time + 10  # 10ms network latency

    expected_frontend_service_rate = [1000/(bs + frontend_process_time) for bs in expected_backend_process_time]
    expected_response_time = [bs + frontend_process_time for bs in expected_backend_process_time]

    # For debug
    # print("[DEBUG] M/G/1/FCFS predicted TPU response time", expected_backend_process_time)

    is_feasible_load = all([lam < mu*FRONTEND_THRESHOLD for mu, lam in zip(expected_frontend_service_rate, input_rate)])
    is_feasible_sla = all([response_time < s for response_time, s in zip(expected_response_time, sla)])

    if not is_feasible_load or not is_feasible_sla:
        LOGGER.info("{} is NOT feasible, load feasible: {}\tSLA feasible: {}".format(
            node.node_name, is_feasible_load, is_feasible_sla))

        return False
    else:
        LOGGER.info("{} is feasible for {} with input rate {:.2f}, result rho: {:.2f}".format(
            node.node_name, frontend_service.service_name, backend_service.input_rates[0], system_rho))
        return True


def utiliztion_diff_tpu(node, new_service):
    """ Compute the utilization different before and after placing a service to a node """
    backend_service = get_tpu_backend_service(node)

    if not backend_service:
        before_util = 0
        service_time = []
        switch_overhead = []
        input_rate = []
        exist_model_index = -1
    else:
        service_time = [1/mu for mu in backend_service.service_rates]
        switch_overhead = [o for o in backend_service.switch_overhead]
        input_rate = [lamb for lamb in backend_service.input_rates]
        before_util = sum(get_mg1_fcfs_response_time(service_time, switch_overhead, input_rate)[1])

        exist_model_index = model_index(new_service, backend_service)

    if exist_model_index == -1:
        service_time.append(1/new_service.service_rates[0])
        switch_overhead.append(new_service.switch_overhead[0])
        input_rate.append(new_service.input_rates[0])
    else:
        input_rate[exist_model_index] += new_service.input_rates[0]

    after_util = sum(get_mg1_fcfs_response_time(service_time, switch_overhead, input_rate)[1])

    return after_util - before_util


def choose_node_breath_first(feasible_nodes: List, new_service):
    """
    Choose a feasible node based on breath first grouping policy:

        1. Search same existing models over the feasible nodes, if found, return that node
        2. If no same existing model, pick an idle node
        3. If no idle node, pick the node with the lowest utilization
    Args:
        feasible_nodes: A list of feasible nodes
        new_service: new service to be placed to the cluster

    Returns:
        node: chosen node
    """

    if not feasible_nodes:
        return None

    # Return the first node that has same model with the new service
    for node in feasible_nodes:
        backend_service = get_tpu_backend_service(node)
        if backend_service:
            index = model_index(new_service, backend_service)

            if index != -1:
                LOGGER.info("Choosing node {} for existing model".format(node.node_name))
                return node

    # Otherwise, place it on an idle node
    for node in feasible_nodes:
        if not get_tpu_backend_service(node):
            LOGGER.info("Choosing node {} for idle node".format(node.node_name))
            return node

    # If no idle nodes, pick the lowest utilized node
    backend_services = [get_tpu_backend_service(n) for n in feasible_nodes]
    utilization = [sum(get_mg1_fcfs_response_time([1/mu for mu in service.service_rates],
                                                  service.switch_overhead,
                                                  service.input_rates)[1]) for service in backend_services]
    sort_index = np.argsort(utilization)[0]
    LOGGER.info("Choosing node {} for least loaded node".format(feasible_nodes[sort_index].node_name))
    return feasible_nodes[sort_index]


def choose_node_depth_first(feasible_nodes: List):
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

    # Sort nodes based on names
    node_names = [n.node_name for n in feasible_nodes]
    index = np.argsort(node_names)[0]

    # Pick first node
    node = feasible_nodes[index]

    LOGGER.info("Choosing node {} using depth first policy".format(node.node_name))
    return node


def model_index(new_service, exist_service):
    """
    Check if new service exist, if exist, return the index of the service in existing backend.
    Otherwise, return  -1
    Args:
        new_service: incoming service
        exist_service: exisiting service

    Returns:
        index
    """
    # new service should have only one model
    new_service_md5 = new_service.model_md5[0]
    index = exist_service.model_md5.index(new_service_md5) if new_service_md5 in exist_service.model_md5 else -1
    return index

