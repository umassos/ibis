#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/4/20
# Description: Placement service
# ========================================================

import os
import csv
import numpy as np
import logging
import hashlib
import grpc
import placement_algorithms_edgetpu

import frontend_service_edgetpu_pb2
import frontend_service_edgetpu_pb2_grpc

from kubernetes import client, config
from collections import namedtuple
from typing import List
from control_service_base import Node, Service, num_generator, copy_to_host, k8s_deploy, k8s_update_deployment

ModelConfigTPU = namedtuple("ModelConfigTPU",
                            ["model_dir", "model_filename", "model_type", "input_rate", "service_rate",
                             "switch_overhead", "frontend_memory", "backend_memory", "frontend_image", "backend_image",
                             "sla", "input_shape"])

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


class EdgeTPUNode(Node):
    """ Node runs EdgeTPU services """
    def get_service_status(self):
        """
        Return the runtime status of the frontend services:

            1. Service name
            2. Input rate
            3. Mean response time
            4. SLO
            5. Expected input rate
        Returns:
            stats: A dict of above features
        """
        stats = {}
        for name, service in self.frontend_services.items():
            ip_address = service.exposed_service.spec.cluster_ip + ':8080'
            with grpc.insecure_channel(ip_address) as channel:
                stub = frontend_service_edgetpu_pb2_grpc.FrontEndServiceTPUStub(channel)

                req = frontend_service_edgetpu_pb2.GetStatus(num_requests=20)
                res = stub.get_status(req)

                stats[name] = {
                    "runtime_input_rate": res.input_rate,
                    "expected_input_rate": service.input_rate,
                    "response_time": res.response_time,
                    "sla": service.sla
                }

        return stats

    def __str__(self, status=dict()):
        """ Print node info """
        pattern = "{:25}: {}\t{}\t{}\n"
        res = "\n"
        res += pattern.format("Node name", self.node_name, '', '')
        res += pattern.format("Ip address", self.ip_address, '', '')
        res += pattern.format("Memory capacity", self.total_memory, '', '')

        if not self.frontend_services:
            res += pattern.format("Frontend services", "", '', '')
        else:
            res += pattern.format("Frontend services", '', '', '')
            for name in self.frontend_services.keys():
                input_rate = status[name]["runtime_input_rate"] if name in status else "NA"
                response_time = status[name]["response_time"] if name in status else "NA"
                res += pattern.format("", name, "lambda={:.5}".format(input_rate),
                                      "r={:.5}".format(response_time))

        if not self.backend_services:
            res += pattern.format("Backend services", "", '', '')
        else:
            service_names = list(self.backend_services.keys())
            res += pattern.format("Backend services", service_names[0], '', '')
            for i in range(1, len(service_names)):
                res += "{:26} {}\n".format("", service_names[i], '', '')

        return res


class FrontendServiceTPU(Service):
    """ Frontend service object """
    def __init__(self, service_name: str, node: Node, memory: int, sla: float, docker_image: str,
                 input_rate: float, backend_service: Service, model_filename: str, input_shape: str,
                 workload_type: str):
        """
        Initialization
        Args:
            service_name: name of this service
            node: node placement
            memory: the amount of memory it used
            sla: service-level agreement
            docker_image: docker image running the service
            input_rate: number of inputs per second
            backend_service: backend service object
            input_shape : str. Shape of the input tensor. Dimensions are separated by 'x'
            workload_type: Type of workload, now only support 'detection' and 'classification'
        """
        super(FrontendServiceTPU, self).__init__(service_name, node, memory, docker_image)
        self.input_rate = input_rate
        self.backend_service = backend_service
        self.model_filename = model_filename
        self.dev_type = 'cpu'

        # Extract model name
        self.model_name = os.path.splitext(self.model_filename)[0]

        self.sla = sla
        self.input_shape = input_shape
        self.workload_type = workload_type

    def get_k8s_deployment_object(self):
        """ Add backend service name """

        assert self.backend_service, "[ERROR] Backend service not specified."
        assert self.backend_service.exposed_service, "[ERROR] Backend service not exposed"
        backend_ip = self.backend_service.exposed_service.spec.cluster_ip
        backend_port = self.backend_service.exposed_service.spec.ports[0].port

        deployment = super(FrontendServiceTPU, self).get_k8s_deployment_object()

        env_vars = [client.V1EnvVar(name="BACKEND_SERVICE_IP", value=backend_ip),
                    client.V1EnvVar(name="BACKEND_SERVICE_PORT", value=str(backend_port)),
                    client.V1EnvVar(name="INPUT_SHAPE", value=self.input_shape),
                    client.V1EnvVar(name="PYTHONUNBUFFERED", value='1'),  # To get logs from pod
                    client.V1EnvVar(name="MODEL_NAME", value=self.model_name),
                    client.V1EnvVar(name="MODEL_TYPE", value=self.workload_type),
                    client.V1EnvVar(name="INPUT_RATE", value=str(self.input_rate))
                    ]

        for container in deployment.spec.template.spec.containers:
            container.env = env_vars

        # The frontend container should have no access to the GPU
        # deployment.spec.template.spec.runtime_class_name = "runc"

        return deployment


class BackendServiceTPU(Service):
    """ Backend service object """

    def __init__(self, service_name: str, node: Node, memory: int, sla: List[float], docker_image: str,
                 input_rates: List[float], service_rates: List[float], switch_overhead: List[float], model_dir: str,
                 model_filenames: List[str], model_types: List[str]):
        """
        Intiailization
        Args:
            service_name: name of this service
            node: node placement
            memory: amount of memory used
            sla: service-level agreement, for each model in model_filenames
            docker_image: docker image running the service
            input_rates: number of inputs per second, for each model in model_filenames
            service_rates: number of inputs processed per second, for each model in model_filenames
            switch_overhead: switch overhead in ms
            model_dir: model directory, separated from path so that all models have to be in the same directorym
            model_filenames: List of .tflite model filename, basename only
            model_types: types of models, 'detection' | 'classification'
        """
        super(BackendServiceTPU, self).__init__(service_name, node, memory, docker_image)
        self.input_rates = input_rates
        self.service_rates = service_rates
        self.switch_overhead = switch_overhead
        self.model_dir = model_dir
        self.model_filenames = model_filenames
        self.model_types = model_types
        self.sla = sla
        self.frontend_services = {}
        self.dev_type = 'tpu'

        # Use MD5 to check if two models are the same
        self.model_md5 = [hashlib.md5(open(os.path.join(self.model_dir, filename), 'rb').read()).hexdigest()
                          for filename in self.model_filenames]

    def get_k8s_deployment_object(self) -> client.V1Deployment:
        """ Add volume config """
        deployment = super(BackendServiceTPU, self).get_k8s_deployment_object()

        # Create security context to allow privileged mode
        security_context = client.V1SecurityContext(privileged=True)

        volume_name = "model-volume"
        volume_mount = client.V1VolumeMount(mount_path="/models", name=volume_name)
        dev_volume_name = "device-volume"
        dev_volume = client.V1VolumeMount(mount_path="/dev/bus/usb", name=dev_volume_name)

        # Add environment variables
        env_vars = [client.V1EnvVar(name="MODELS", value=",".join(self.model_filenames)),
                    client.V1EnvVar(name="MODEL_TYPES", value=",".join(self.model_types)),
                    client.V1EnvVar(name="PYTHONUNBUFFERED", value='1')  # to get logs from the pod
                    ]

        for container in deployment.spec.template.spec.containers:
            container.volume_mounts = [volume_mount, dev_volume]
            container.env = env_vars
            container.security_context = security_context

        # Mount model directory to container
        pod_volume = client.V1Volume(name=volume_name)
        pod_volume.host_path = client.V1HostPathVolumeSource(self.model_dir)

        # Mount the use deivce to container
        pod_dev_volume = client.V1Volume(name=dev_volume_name)
        pod_dev_volume.host_path = client.V1HostPathVolumeSource("/dev/bus/usb")

        deployment.spec.template.spec.volumes = [pod_volume, pod_dev_volume]

        return deployment


class ControlServiceTPU(Service):
    """ Control service class """
    def __init__(self, method="bfs"):
        """ Get nodes information from K8s """
        config.load_kube_config()
        self.k8s_v1 = client.CoreV1Api()
        self.k8s_apps_v1 = client.AppsV1Api()

        assert "CONTROL_SERVICE_HOSTNAME" in os.environ, "Enviroment variable 'CONTROL_SERVICE_HOSTNAME' has to be set"
        hostname = os.environ["CONTROL_SERVICE_HOSTNAME"]

        # Get current node
        selector = "kubernetes.io/hostname=%s" % hostname
        current_node = Node(self.k8s_v1.list_node(label_selector=selector).items[0])


        #######################################################################################
        # TODO: default control service allocated memory is set to 3 Gb, change this if needed
        #######################################################################################
        memory = 3072
        super(ControlServiceTPU, self).__init__("control-service", current_node, memory, docker_image="")

        # Get nodes
        self.nodes = {}
        k8s_nodes = self.k8s_v1.list_node()

        for node in k8s_nodes.items:
            new_node = EdgeTPUNode(node)

            # Control is not included
            if new_node.node_name != self.node.node_name:
                self.nodes[new_node.node_name] = new_node

        # Gen suffix generator
        self.suffix_generator = num_generator()

        LOGGER.info("Using {} placement policy...".format(method))
        self.method = method

    def get_schedule_node(self, frontend_service: FrontendServiceTPU, backend_service: BackendServiceTPU) -> Node:
        """
        Find a feasible node for both frontend and backend services.
        Args:
            frontend_service: frontend service
            backend_service: backend service
            modeling: if use our queuing model to choose node

        Returns:
            node: Node object
        """
        feasible_nodes = []

        if self.method == "bfs":
            for _, node in self.nodes.items():
                if placement_algorithms_edgetpu.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_breath_first(feasible_nodes, backend_service)
        elif self.method == "dfs":
            for _, node in self.nodes.items():
                if placement_algorithms_edgetpu.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_depth_first(feasible_nodes)
        elif self.method == "grouping":
            for _, node in self.nodes.items():
                if node.node_name not in placement_algorithms_edgetpu.INFEASIBLE_NODES:
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_breath_first(feasible_nodes, backend_service)
        else:
            raise ValueError("Unknown placement method {}".format(self.method))

    def deploy(self, model_config: ModelConfigTPU):
        """
        Deploy a model to the cluster
        Args:
            model_config: data structure to store model configuration

        Returns:

        """
        suffix = next(self.suffix_generator)
        model_name = os.path.splitext(model_config.model_filename)[0]
        frontend_service_name = model_name.lower() + "-frontend-" + suffix
        frontend_service_name = frontend_service_name.replace('_', '-')   # '_' cannot be used in domain name
        backend_service_name = "model-backend-" + suffix

        # Define frontend service
        frontend_service = FrontendServiceTPU(service_name=frontend_service_name, node=None,
                                              memory=model_config.frontend_memory, sla=model_config.sla,
                                              docker_image=model_config.frontend_image, backend_service=None,
                                              model_filename=model_config.model_filename,
                                              input_rate=model_config.input_rate,
                                              input_shape=model_config.input_shape,
                                              workload_type=model_config.model_type)

        # Define backend service
        backend_service = BackendServiceTPU(service_name=backend_service_name, node=None,
                                            memory=model_config.backend_memory, sla=[model_config.sla],
                                            docker_image=model_config.backend_image,
                                            model_dir=model_config.model_dir,
                                            input_rates=[model_config.input_rate],
                                            service_rates=[model_config.service_rate],
                                            switch_overhead=[model_config.switch_overhead],
                                            model_filenames=[model_config.model_filename],
                                            model_types=[model_config.model_type])

        # Placement algorithm happens here
        target_node = self.get_schedule_node(frontend_service, backend_service)

        if not target_node:
            print("[WARNING] No nodes feasible..")
            return None, None
        else:
            # If there are already some models loaded on the node, add the new model to the list,
            # then update the Pod
            is_service_exist = len(target_node.backend_services) > 0
            exist_model_index = -1

            if is_service_exist:
                # EdgeTPU device cannot be shared across processes
                assert len(target_node.backend_services) == 1, "[ERROR] Only one EdgeTPU program can run on one node. "

                exist_backend_service_name, exist_backend_service = next(iter(target_node.backend_services.items()))
                exist_model_index = placement_algorithms_edgetpu.model_index(backend_service, exist_backend_service)

                # Append new model configuration to the exist list
                assert backend_service.model_dir == exist_backend_service.model_dir, \
                    "[ERROR] All models have to be in the same directory, but %s != %s" %\
                    (backend_service.model_dir, exist_backend_service.model_dir)
                assert backend_service.docker_image == exist_backend_service.docker_image, \
                    "[ERROR] Backend containers should be the same, but got %s != %s" %\
                    (backend_service.docker_image, exist_backend_service.docker_image)

                if exist_model_index == -1:
                    # If incoming model not exist on the node, append it
                    exist_backend_service.input_rates.append(backend_service.input_rates[0])
                    exist_backend_service.service_rates.append(backend_service.service_rates[0])
                    exist_backend_service.switch_overhead.append(backend_service.switch_overhead[0])
                    exist_backend_service.model_filenames.append(backend_service.model_filenames[0])
                    exist_backend_service.model_types.append(backend_service.model_types[0])
                    exist_backend_service.sla.append(backend_service.sla[0])
                    exist_backend_service.memory += backend_service.memory
                    exist_backend_service.model_md5.append(backend_service.model_md5[0])
                else:
                    # If incoming mode has already existed, just use the existing one
                    exist_backend_service.input_rates[exist_model_index] += backend_service.input_rates[0]
                    exist_backend_service.sla[exist_model_index] = min(backend_service.sla[0],
                                                                       exist_backend_service.sla[exist_model_index])

                backend_service = exist_backend_service

            # Config frontend service
            frontend_service.node = target_node
            frontend_service.backend_service = backend_service
            frontend_service.log_requiest = []

            # Config backend service
            backend_service.node = target_node
            backend_service.frontend_services[frontend_service.service_name] = frontend_service

            # Copy engine to host
            for filename in backend_service.model_filenames:
                copy_to_host(os.path.join(backend_service.model_dir, filename), backend_service.node)

            # Deploy services
            if is_service_exist and exist_model_index == -1:
                # Add new model to backend container
                k8s_update_deployment(self.k8s_apps_v1, backend_service)
            elif not is_service_exist:
                # Create new backend container
                k8s_deploy(self.k8s_apps_v1, self.k8s_v1, backend_service)

            # Create frontend container
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, frontend_service)

            # register to nodes
            frontend_service.node.frontend_services[frontend_service.service_name] = frontend_service
            backend_service.node.backend_services[backend_service.service_name] = backend_service

            return frontend_service, backend_service

    def detect_hotspot(self):
        """
        Detect SLO violation and do migration. Now only support migrate to an idle node
        Returns:

        """
        for node_name, node in self.nodes.items():
            node_status = node.get_service_status()
            for service_name, service_status in node_status.items():
                if service_status["response_time"] > service_status["sla"]:
                    LOGGER.info("Detect SLO violation for {} on {}, {:.2f} > {:.2f}, perform migration"
                                .format(service_name, node_name, service_status["response_time"],
                                        service_status["sla"]))
                    # LOGGER.info("Detect input rate boost for {} on {}, {:.2f} > {:.2f}, perform migration"
                    #             .format(service_name, node_name, service_status["runtime_input_rate"],
                    #                     service_status["expected_input_rate"]))

                    # 1. Remove service from the node
                    migrate_frontend = node.frontend_services.pop(service_name)
                    migrate_frontend.input_rate = service_status["runtime_input_rate"]
                    source_backend = node.backend_services[migrate_frontend.backend_service.service_name]

                    index = source_backend.model_filenames.index(migrate_frontend.model_filename)

                    source_backend.input_rates.pop(index)
                    source_backend.model_md5.pop(index)
                    # new_backend.input_rates = [service_status["runtime_input_rate"]]
                    # new_backend.sla = [source_backend.sla.pop(index)]
                    # new_backend.service_rates = [source_backend.service_rates.pop(index)]
                    # new_backend.switch_overhead = [source_backend.switch_overhead.pop(index)]
                    # new_backend.model_filenames = [source_backend.model_filenames.pop(index)]
                    # new_backend.model_types = [source_backend.model_typles.pop(index)]
                    # new_backend.service_name = new_backend.service_name + "-migrate"
                    # new_backend.model_md5 = [source_backend.model_md5.pop(index)]

                    new_backend = BackendServiceTPU(service_name=source_backend.service_name + "-migrate",
                                                    node=None,
                                                    memory=source_backend.memory,
                                                    sla=[source_backend.sla.pop(index)],
                                                    docker_image=source_backend.docker_image,
                                                    model_dir=source_backend.model_dir,
                                                    input_rates=[service_status["runtime_input_rate"]],
                                                    service_rates=[source_backend.service_rates.pop(index)],
                                                    switch_overhead=[source_backend.switch_overhead.pop(index)],
                                                    model_filenames=[source_backend.model_filenames.pop(index)],
                                                    model_types=[source_backend.model_types.pop(index)])

                    target_node = self.get_schedule_node(migrate_frontend, new_backend)
                    if target_node.backend_services:
                        LOGGER.error("Node {} already has something running".format(target_node.node_name))
                        raise RuntimeError()

                    # Configure frontend
                    migrate_frontend.node = target_node
                    migrate_frontend.backend_service = new_backend

                    # config backend
                    new_backend.node = target_node
                    new_backend.frontend_services = {service_name: migrate_frontend}

                    LOGGER.info("Migrate {} from {} to {}".format(service_name, node_name, target_node.node_name))

                    # Copy engine to host
                    for filename in new_backend.model_filenames:
                        copy_to_host(os.path.join(new_backend.model_dir, filename), new_backend.node)

                    k8s_deploy(self.k8s_apps_v1, self.k8s_v1, new_backend)
                    k8s_update_deployment(self.k8s_apps_v1, migrate_frontend)

                    # register to nodes
                    migrate_frontend.node.frontend_services[migrate_frontend.service_name] = migrate_frontend
                    new_backend.node.backend_services[new_backend.service_name] = new_backend

                    # Move one at a time
                    return

    def destroy(self, save_path="test.csv"):
        stats = []
        for _, node in self.nodes.items():
            for service_name, service in node.frontend_services.items():
                print("[INFO] Deleting Deployment %s" % service_name)
                self.k8s_apps_v1.delete_namespaced_deployment(service_name, "default")

                if service.exposed_service:
                    print("[INFO] Deleting Service %s " % service_name)
                    self.k8s_v1.delete_namespaced_service(name=service_name, namespace="default")

            for service_name, service in node.backend_services.items():
                print("[INFO] Deleting Deployment %s" % service_name)
                self.k8s_apps_v1.delete_namespaced_deployment(service_name, "default")

                if service.exposed_service:
                    print("[INFO] Deleting Service %s " % service_name)
                    self.k8s_v1.delete_namespaced_service(name=service_name, namespace="default")

                # Collect backend information
                service_time = [1 / mu for mu in service.service_rates]
                expected_response_time, expected_rho = \
                    placement_algorithms_edgetpu.get_mg1_fcfs_response_time(service_time,
                                                                            service.switch_overhead,
                                                                            service.input_rates)

                for filename, predicted_time, input_rate, rho, sla in \
                        zip(service.model_filenames, expected_response_time, service.input_rates,
                            expected_rho, service.sla):
                    stats.append({
                        "model_filename": filename,
                        "node": node.node_name,
                        "expected_inference_time": predicted_time,
                        "expected_preprocess_time": placement_algorithms_edgetpu.get_mgc_ps_response_time(input_rate),
                        "input_rate": input_rate,
                        "rho": rho,
                        "sla": sla
                    })

        keys = stats[0].keys()
        with open(save_path, 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats)
