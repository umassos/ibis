#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 1/4/21
# Description: 
# ========================================================

import os
import csv
import logging
import random
import numpy as np
import placement_algorithms_trt
import placement_algorithms_edgetpu

from kubernetes import client, config
from collections import namedtuple
from control_service_base import Node, Service, num_generator, k8s_deploy, copy_to_host, k8s_update_deployment
from control_service_edgetpu import FrontendServiceTPU, BackendServiceTPU
from control_service_trt import FrontendServiceGPU, BackendServiceGPU

ModelConfig = namedtuple("ModelConfig",
                         ["model_dir_tpu", "model_filename_tpu", "model_type_tpu", "input_rate", "service_rate_tpu",
                          "switch_overhead", "frontend_memory_tpu", "backend_memory_tpu", "frontend_image_tpu",
                          "backend_image_tpu", "sla", "input_shape_tpu", "model_path_gpu", "model_name",
                          "output_bindings",
                          "deployment_type", "frontend_image_gpu", "backend_image_gpu", "service_rate_gpu",
                          "frontend_memory_gpu", "backend_memory_gpu", "input_shape_gpu", "target_dev"])

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


class ControlService(Service):
    """ Control service for GPU and TPU cluster """
    def __init__(self, tpu_method="bfs", gpu_method="modeling_dfs", dev_method="best"):
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
        super(ControlService, self).__init__("control-service", current_node, memory, docker_image="")

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

        LOGGER.info("Using {} placement policy for TPU, {} for GPU".format(tpu_method, gpu_method))
        self.tpu_method = tpu_method
        self.gpu_method = gpu_method
        self.dev_method = dev_method

    def _get_schedule_node_tpu(self, frontend_service: FrontendServiceTPU, backend_service: BackendServiceTPU) -> Node:
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

        if self.tpu_method == "bfs":
            for _, node in self.nodes.items():
                if placement_algorithms_edgetpu.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_breath_first(feasible_nodes, backend_service)
        elif self.tpu_method == "dfs":
            for _, node in self.nodes.items():
                if placement_algorithms_edgetpu.is_node_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_depth_first(feasible_nodes)
        elif self.tpu_method == "grouping":
            for _, node in self.nodes.items():
                if node.node_name not in placement_algorithms_edgetpu.INFEASIBLE_NODES and\
                        placement_algorithms_edgetpu.is_memory_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)

            return placement_algorithms_edgetpu.choose_node_breath_first(feasible_nodes, backend_service)
        else:
            raise ValueError("Unknown placement method {}".format(self.tpu_method))

    def _get_schedule_node_gpu(self, frontend_service: FrontendServiceGPU, backend_service: BackendServiceGPU) -> Node:
        """
        Find a feasible node for both frontend and backend services.
        Args:
            frontend_service: frontend service
            backend_service: backend service

        Returns:
            node: Node object
        """
        feasible_nodes = []

        if self.gpu_method == "modeling_dfs":
            for _, node in self.nodes.items():
                if backend_service.model_type == "MAAS":
                    is_feasible = placement_algorithms_trt.is_node_feasible_MaaS(frontend_service,
                                                                                 backend_service, node)
                else:
                    is_feasible = placement_algorithms_trt.is_node_feasible(frontend_service, backend_service, node)

                if is_feasible:
                    feasible_nodes.append(node)

            return placement_algorithms_trt.choose_node_depth_first(feasible_nodes, backend_service)
        elif self.gpu_method == "modeling_bfs":
            for _, node in self.nodes.items():
                if backend_service.model_type == "MAAS":
                    is_feasible = placement_algorithms_trt.is_node_feasible_MaaS(frontend_service, backend_service,
                                                                                 node)
                else:
                    is_feasible = placement_algorithms_trt.is_node_feasible(frontend_service, backend_service, node)

                if is_feasible:
                    feasible_nodes.append(node)

            return placement_algorithms_trt.choose_node_breadth_first(feasible_nodes, backend_service)
        elif self.gpu_method.split("_")[0] == "memory":
            for _, node in self.nodes.items():
                if node.node_name not in placement_algorithms_trt.INFEASIBLE_NODES and \
                        placement_algorithms_trt.is_memory_feasible(frontend_service, backend_service, node):
                    feasible_nodes.append(node)
            return placement_algorithms_trt.choose_node_memory(feasible_nodes, backend_service, self.gpu_method)
        else:
            raise ValueError("Unknown placement method {}".format(self.gpu_method))

    def _find_feasible_node_tpu(self, model_config: ModelConfig):
        """ Search a feasible TPU node """
        suffix = next(self.suffix_generator)
        model_name = os.path.splitext(model_config.model_filename_tpu)[0]
        frontend_service_name = model_name.lower() + "-frontend-" + suffix
        frontend_service_name = frontend_service_name.replace('_', '-')  # '_' cannot be used in domain name
        backend_service_name = "tpu-backend-" + suffix

        # Define frontend service
        frontend_service = FrontendServiceTPU(service_name=frontend_service_name, node=None,
                                              memory=model_config.frontend_memory_tpu, sla=model_config.sla,
                                              docker_image=model_config.frontend_image_tpu, backend_service=None,
                                              model_filename=model_config.model_filename_tpu,
                                              input_rate=model_config.input_rate,
                                              input_shape=model_config.input_shape_tpu,
                                              workload_type=model_config.model_type_tpu)

        # Define backend service
        backend_service = BackendServiceTPU(service_name=backend_service_name, node=None,
                                            memory=model_config.backend_memory_tpu, sla=[model_config.sla],
                                            docker_image=model_config.backend_image_tpu,
                                            model_dir=model_config.model_dir_tpu,
                                            input_rates=[model_config.input_rate],
                                            service_rates=[model_config.service_rate_tpu],
                                            switch_overhead=[model_config.switch_overhead],
                                            model_filenames=[model_config.model_filename_tpu],
                                            model_types=[model_config.model_type_tpu])

        return self._get_schedule_node_tpu(frontend_service, backend_service), frontend_service, backend_service

    def _find_feasible_node_gpu(self, model_config: ModelConfig):
        """ Search a feasible GPU node """
        suffix = next(self.suffix_generator)
        frontend_service_name = model_config.model_name.lower() + "-frontend-" + suffix
        frontend_service_name = frontend_service_name.replace('_', '-')  # '_' cannot be used in domain name
        backend_service_name = "gpu-backend-" + suffix

        # Define frontend service
        frontend_service = FrontendServiceGPU(service_name=frontend_service_name, node=None,
                                              memory=model_config.frontend_memory_gpu, sla=model_config.sla,
                                              docker_image=model_config.frontend_image_gpu, backend_service=None,
                                              model_path=model_config.model_path_gpu, input_rate=model_config.input_rate,
                                              input_shape=model_config.input_shape_gpu,
                                              output_bindings=model_config.output_bindings)

        # Define backend service
        backend_service = BackendServiceGPU(service_name=backend_service_name, node=None,
                                            memory=model_config.backend_memory_gpu, sla=model_config.sla,
                                            docker_image=model_config.backend_image_gpu,
                                            model_name=model_config.model_name,
                                            model_path=model_config.model_path_gpu, engine_path="",
                                            input_rate=model_config.input_rate,
                                            service_rate=model_config.service_rate_gpu,
                                            input_shape=model_config.input_shape_gpu, wait_time=0,
                                            model_type=model_config.deployment_type)

        return self._get_schedule_node_gpu(frontend_service, backend_service), frontend_service, backend_service

    def _deply_to_node_tpu(self, target_node, frontend_service, backend_service):
        """ Deploy the frontend and backend service to target node """
        # If there are already some models loaded on the node, add the new model to the list,
        # then update the Pod
        exist_backend_service = placement_algorithms_edgetpu.get_tpu_backend_service(target_node)
        exist_model_index = -1

        if exist_backend_service:
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
        # frontend_service.log_request = []

        # Config backend service
        backend_service.node = target_node
        backend_service.frontend_services[frontend_service.service_name] = frontend_service

        # Copy engine to host
        # for filename in backend_service.model_filenames:
        #     copy_to_host(os.path.join(backend_service.model_dir, filename), backend_service.node)

        # Deploy services
        if exist_backend_service and exist_model_index == -1:
            # Add new model to backend container
            k8s_update_deployment(self.k8s_apps_v1, backend_service)
        elif not exist_backend_service:
            # Create new backend container
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, backend_service)

        # Create frontend container
        k8s_deploy(self.k8s_apps_v1, self.k8s_v1, frontend_service)

        # register to nodes
        frontend_service.node.frontend_services[frontend_service.service_name] = frontend_service
        backend_service.node.backend_services[backend_service.service_name] = backend_service
        return frontend_service, backend_service

    def _deply_to_node_gpu(self, target_node, frontend_service, backend_service):
        """ Deploy the frontend and backend service to target node """
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
            # print("Sending ", backend_service.engine_path)
            # copy_to_host(backend_service.engine_path, backend_service.node)

            # Deploy services
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, backend_service)
            k8s_deploy(self.k8s_apps_v1, self.k8s_v1, frontend_service)

            # register to nodes
            frontend_service.node.frontend_services[frontend_service.service_name] = frontend_service
            backend_service.node.backend_services[backend_service.service_name] = backend_service

            return frontend_service, backend_service

    def deploy(self, model_config: ModelConfig):
        """
        Deploy a model to the cluster
        Args:
            model_config: model configuration

        Returns:

        """
        # find a TPU node
        target_tpu_node, tpu_frontend, tpu_backend = self._find_feasible_node_tpu(model_config)

        # find a GPU node
        target_gpu_node, gpu_frontend, gpu_backend = self._find_feasible_node_gpu(model_config)

        if not target_tpu_node and target_gpu_node or model_config.target_dev == 'gpu':
            LOGGER.info("No TPU device available or GPU specified, choose GPU.")
            return self._deply_to_node_gpu(target_gpu_node, gpu_frontend, gpu_backend)

        elif target_tpu_node and not target_gpu_node or model_config.target_dev == 'tpu':
            LOGGER.info("No GPU device available or TPU specified, choose TPU")
            return self._deply_to_node_tpu(target_tpu_node, tpu_frontend, tpu_backend)
        elif target_tpu_node and target_gpu_node:
            if self.dev_method == 'best':
                tpu_util_delta = placement_algorithms_edgetpu.utiliztion_diff_tpu(target_tpu_node, tpu_backend)
                gpu_util_delta = placement_algorithms_trt.utilization_diff_gpu(target_gpu_node, gpu_backend)

                if tpu_util_delta < gpu_util_delta:
                    LOGGER.info("TPU util {:.2f} is smaller than {:.2f} of GPU, choose TPU".format(tpu_util_delta,
                                                                                                 gpu_util_delta))
                    return self._deply_to_node_tpu(target_tpu_node, tpu_frontend, tpu_backend)
                else:
                    LOGGER.info("GPU util {:.2f} is smaller than {:.2f} of TPU, choose GPU".format(gpu_util_delta,
                                                                                                 tpu_util_delta))
                    return self._deply_to_node_gpu(target_gpu_node, gpu_frontend, gpu_backend)
            else:
                # Choose randomly
                if np.random.randint(2) == 0:
                    LOGGER.info("Randomly choose TPU")
                    return self._deply_to_node_tpu(target_tpu_node, tpu_frontend, tpu_backend)
                else:
                    LOGGER.info("Randomly choose GPU")
                    return self._deply_to_node_gpu(target_gpu_node, gpu_frontend, gpu_backend)
        else:
            LOGGER.warning("No nodes available")
            return None, None

    def destroy(self, save_path="test.csv"):
        """ Destroy the deployments """
        stats = []
        for _, node in self.nodes.items():
            for service_name, service in node.frontend_services.items():
                print("[INFO] Deleting Deployment %s" % service_name)
                self.k8s_apps_v1.delete_namespaced_deployment(service_name, "default")

                if service.exposed_service:
                    print("[INFO] Deleting Service %s " % service_name)
                    self.k8s_v1.delete_namespaced_service(name=service_name, namespace="default")

            # Delete TPU backend
            service = placement_algorithms_edgetpu.get_tpu_backend_service(node)

            if service:
                service_name = service.service_name
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
                        "expected_response_time": predicted_time,
                        "expected_preprocess_time": placement_algorithms_edgetpu.get_mgc_ps_response_time(input_rate),
                        "input_rate": input_rate,
                        "rho": rho,
                        "sla": sla,
                        "dev": 'tpu'
                    })

            # Collect GPU result
            service_time = []
            sla = []
            input_rate = []
            model_names = []

            for service_name, service in placement_algorithms_trt.get_gpu_backend_services(node).items():
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
            expected_inference_time, expected_rho =\
                placement_algorithms_trt.get_mg1_ps_response_time(service_time, input_rate)

            for i in range(len(model_names)):
                stats.append({
                        "model_filename": model_names[i],
                        "node": node.node_name,
                        "expected_response_time": expected_inference_time[i],
                        "expected_preprocess_time": placement_algorithms_trt.get_mgc_ps_response_time(input_rate[i])[1],
                        "input_rate": input_rate[i],
                        "rho": expected_rho[i],
                        "sla": sla[i],
                        "dev": 'gpu'
                    })

        keys = stats[0].keys()
        with open(save_path, 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats)