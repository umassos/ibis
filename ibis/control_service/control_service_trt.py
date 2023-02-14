#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/4/20
# Description: Placement service
# ========================================================

import os
import csv
import logging
import placement_algorithms_trt

from kubernetes import client, config
from collections import namedtuple
from control_service_base import Node, Service, num_generator, k8s_deploy, copy_to_host

ModelConfigGPU = namedtuple("ModelConfigGPU", ["model_name", "model_path", "input_rate", "service_rate",
                                               "frontend_memory", "backend_memory", "frontend_image", "backend_image",
                                               "sla", "input_shape", "output_bindings", "wait_time", "model_type"])

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


class FrontendServiceGPU(Service):
    """ Frontend service object """

    def __init__(self, service_name: str, node: Node, memory: int, sla: float, docker_image: str,
                 input_rate: float, backend_service: Service, model_path: str, input_shape: str,
                 output_bindings: str):
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
            model_path: path to the model file
            input_shape : str. Shape of the input tensor. Dimensions are separated by 'x'
            output_bindings: bindings for output
        """
        super(FrontendServiceGPU, self).__init__(service_name, node, memory, docker_image)
        self.input_rate = input_rate
        self.backend_service = backend_service
        self.model_path = model_path
        self.sla = sla
        self.input_shape = input_shape
        self.output_bindings = output_bindings
        self.dev_type = 'cpu'

    def get_k8s_deployment_object(self):
        """ Add backend service name """

        assert self.backend_service, "[ERROR] Backend service not specified."
        assert self.backend_service.exposed_service, "[ERROR] Backend service not exposed"
        backend_ip = self.backend_service.exposed_service.spec.cluster_ip
        backend_port = self.backend_service.exposed_service.spec.ports[0].port

        deployment = super(FrontendServiceGPU, self).get_k8s_deployment_object()

        env_vars = [client.V1EnvVar(name="BACKEND_SERVICE_IP", value=backend_ip),
                    client.V1EnvVar(name="BACKEND_SERVICE_PORT", value=str(backend_port)),
                    client.V1EnvVar(name="INPUT_SHAPE", value=self.input_shape),
                    client.V1EnvVar(name="PYTHONUNBUFFERED", value='1'),  # To get logs from pod
                    client.V1EnvVar(name="OUTPUT_BINDING", value=self.output_bindings)
                    ]

        for container in deployment.spec.template.spec.containers:
            container.env = env_vars

        # The frontend container should have no access to the GPU
        # deployment.spec.template.spec.runtime_class_name = "runc"

        return deployment


class BackendServiceGPU(Service):
    """ Backend service object """

    def __init__(self, service_name: str, node: Node, memory: int, sla: float, docker_image: str,
                 input_rate: float, service_rate: float, model_name: str, model_path: str, engine_path: str,
                 input_shape: str = "1x3x224x224", wait_time: int = 0, model_type = "user"):
        """
        Intiailization
        Args:
            service_name: name of this service
            node: node placement
            memory: amount of memory used
            sla: service-level agreement
            docker_image: docker image running the service
            input_rate: number of inputs per second
            service_rate: number of inputs processed per second
            model_path: path to .onnx model file
            engine_path: path to TRT engine file
            input_shape : str. Shape of the input tensor. Dimensions are separated by 'x'
            wait_time: the wait time for batching
        """
        super(BackendServiceGPU, self).__init__(service_name, node, memory, docker_image)
        self.input_rate = input_rate
        self.service_rate = service_rate
        self.model_name = model_name
        self.model_path = model_path
        self.engine_path = engine_path
        self.sla = sla
        self.frontend_services = {}
        self.input_shape = input_shape
        self.wait_time = wait_time
        self.model_type = model_type
        self.dev_type = 'gpu'

    def get_k8s_deployment_object(self) -> client.V1Deployment:
        """ Add volume config """
        deployment = super(BackendServiceGPU, self).get_k8s_deployment_object()

        volume_name = "model-volume"
        volume_mount = client.V1VolumeMount(mount_path="/models", name=volume_name)

        # Add environment variables
        env_vars = [client.V1EnvVar(name="INPUT_SHAPE", value=self.input_shape),
                    client.V1EnvVar(name="WAIT_TIME", value=str(self.wait_time)),
                    client.V1EnvVar(name="fid", value=os.path.basename(self.engine_path)),
                    client.V1EnvVar(name="PYTHONUNBUFFERED", value='1')  # to get logs from the pod
                    ]

        for container in deployment.spec.template.spec.containers:
            container.volume_mounts = [volume_mount]
            container.env = env_vars

        pod_volume = client.V1Volume(name=volume_name)
        pod_volume.host_path = client.V1HostPathVolumeSource(os.path.dirname(self.engine_path))

        deployment.spec.template.spec.volumes = [pod_volume]

        return deployment


class ControlServiceGPU(Service):
    """ Control service class """

    def __init__(self, method="modeling_dfs"):
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
        super(ControlServiceGPU, self).__init__("control-service", current_node, memory, docker_image="")

        # Get nodes
        self.nodes = {}
        k8s_nodes = self.k8s_v1.list_node()

        for node in k8s_nodes.items:
            new_node = Node(node)

            # Control is not included
            if new_node.node_name != self.node.node_name:
                self.nodes[new_node.node_name] = new_node

        # Gen suffix generator
        self.suffix_generator = num_generator()
        
        LOGGER.info("Using {} placement policy...".format(method))
        self.method = method

    def get_schedule_node(self, frontend_service: FrontendServiceGPU, backend_service: BackendServiceGPU) -> Node:
        """
        Find a feasible node for both frontend and backend services.
        Args:
            frontend_service: frontend service
            backend_service: backend service

        Returns:
            node: Node object
        """
        feasible_nodes = []

        if self.method == "modeling_dfs":
            for _, node in self.nodes.items():
                if backend_service.model_type == "MAAS" and \
                        placement_algorithms_trt.is_node_feasible_MaaS(frontend_service, backend_service, node):
                    feasible_nodes.append(node)
                elif placement_algorithms_trt.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_trt.choose_node_depth_first(feasible_nodes, backend_service)
        elif self.method == "modeling_bfs":
            for _, node in self.nodes.items():
                if backend_service.model_type == "MAAS" and \
                        placement_algorithms_trt.is_node_feasible_MaaS(frontend_service, backend_service, node):
                    feasible_nodes.append(node)
                elif placement_algorithms_trt.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_trt.choose_node_breadth_first(feasible_nodes, backend_service)
        elif self.method.split("_")[0] == "memory":
            for _, node in self.nodes.items():
                if node.node_name not in placement_algorithms_trt.INFEASIBLE_NODES and \
                        placement_algorithms_trt.is_memory_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)
            return placement_algorithms_trt.choose_node_memory(feasible_nodes, backend_service, self.method)
        else:
            raise ValueError("Unknown placement method {}".format(self.method))

    def deploy(self, model_config: ModelConfigGPU):
        """
        Deploy a model to the cluster
        Args:
            model_config: data structure to store model configuration

        Returns:

        """
        suffix = next(self.suffix_generator)
        frontend_service_name = model_config.model_name.lower() + "-frontend-" + suffix
        frontend_service_name = frontend_service_name.replace('_', '-')   # '_' cannot be used in domain name
        backend_service_name = "model-backend-" + suffix

        # Define frontend service
        frontend_service = FrontendServiceGPU(service_name=frontend_service_name, node=None,
                                              memory=model_config.frontend_memory, sla=model_config.sla,
                                              docker_image=model_config.frontend_image, backend_service=None,
                                              model_path=model_config.model_path, input_rate=model_config.input_rate,
                                              input_shape=model_config.input_shape,
                                              output_bindings=model_config.output_bindings)

        # Define backend service
        backend_service = BackendServiceGPU(service_name=backend_service_name, node=None,
                                            memory=model_config.backend_memory, sla=model_config.sla,
                                            docker_image=model_config.backend_image, 
                                            model_name=model_config.model_name,
                                            model_path=model_config.model_path, engine_path="",
                                            input_rate=model_config.input_rate, service_rate=model_config.service_rate,
                                            input_shape=model_config.input_shape, wait_time=model_config.wait_time,
                                            model_type=model_config.model_type)

        target_node = self.get_schedule_node(frontend_service, backend_service)

        if not target_node:
            print("[WARNING] No nodes feasible..")
            return None, None

        index = placement_algorithms_trt.contains_model(target_node, backend_service)

        if index != -1 and backend_service.model_type == "MAAS":
            current_backend = placement_algorithms_trt.get_needed_service(target_node, backend_service)
            current_backend.input_rate += backend_service.input_rate
            current_backend.sla = min(current_backend.sla,  backend_service.sla)
            # Config frontend service
            frontend_service.node = target_node
            frontend_service.backend_service = current_backend
            # Config Backend add new front end
            current_backend.frontend_services[frontend_service.service_name] = frontend_service

            # Deploy Front End Service
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, frontend_service)

            # register to nodes
            frontend_service.node.frontend_services[frontend_service.service_name] = frontend_service
            current_backend.node.backend_services[current_backend.service_name] = current_backend

            return frontend_service, current_backend
        else:
            # Config frontend service
            frontend_service.node = target_node
            frontend_service.backend_service = backend_service

            # Config backend service
            backend_service.node = target_node
            backend_service.frontend_services[frontend_service.service_name] = frontend_service

            backend_service.engine_path = backend_service.model_path    # Assume engines are built

            # Copy engine to host
            #print("Sending ", backend_service.engine_path)
            #copy_to_host(backend_service.engine_path, backend_service.node)

            # Deploy services
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, backend_service)
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, frontend_service)

            # register to nodes
            frontend_service.node.frontend_services[frontend_service.service_name] = frontend_service
            backend_service.node.backend_services[backend_service_name] = backend_service

            return frontend_service, backend_service

    def destroy(self, save_path="tes.csv"):
        stats = []
        for _, node in self.nodes.items():
            for service_name, service in node.frontend_services.items():
                print("[INFO] Deleting Deployment %s" % service_name)
                self.k8s_apps_v1.delete_namespaced_deployment(service_name, "default")

                if service.exposed_service:
                    print("[INFO] Deleting Service %s " % service_name)
                    self.k8s_v1.delete_namespaced_service(name=service_name, namespace="default")

            service_time = []
            sla = []
            input_rate = []
            model_names = []
            clients = []

            for service_name, service in node.backend_services.items():
                print("[INFO] Deleting Deployment %s" % service_name)
                self.k8s_apps_v1.delete_namespaced_deployment(service_name, "default")

                if service.exposed_service:
                    print("[INFO] Deleting Service %s " % service_name)
                    self.k8s_v1.delete_namespaced_service(name=service_name, namespace="default")

                # Collect backend information
                service_time.append(1/service.service_rate)
                model_names.append(service.model_name)
                sla.append(service.sla)
                input_rate.append(service.input_rate)
                clients.append(len(service.frontend_services))
            expected_inference_time, expected_rho =\
                placement_algorithms_trt.get_mg1_ps_response_time(service_time, input_rate)

            for i in range(len(model_names)):
                stats.append({
                        "model_filename": model_names[i],
                        "node": node.node_name,
                        "expected_response_time": expected_inference_time[i],
                        "expected_preprocess_time": placement_algorithms_trt.get_mgc_ps_response_time(input_rate[i]),
                        "input_rate": input_rate[i],
                        "rho": expected_rho[i],
                        "sla": sla[i],
                        "clients": clients[i]
                    })

        keys = stats[0].keys()
        with open(save_path, 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats)


