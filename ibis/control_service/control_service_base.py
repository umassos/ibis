#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/4/20
# Description: Placement service
# ========================================================

import re
import os
import random
import string
import subprocess

from kubernetes import client


def get_random_string(length=8):
    # Random string with the combination of lower and upper case
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def num_generator():
    """ Generate integer from 0 increasingly """
    num = 0

    while True:
        yield str(num)
        num += 1


def get_number(text: str):
    """
    Get first appear integer number from text
    Args:
        text: input text

    Returns:
        num: integer

    """
    return int(re.findall("\d+", text)[0])


def copy_to_host(engine_path, node):
    """
    Copy engine file to host
    Args:
        engine_path: absolution path to engine file
        node: target node

    Returns:

    """
    dir = os.path.dirname(engine_path)

    mkdir_cmd = ["ssh", "picocluster@%s" % (node.ip_address), "mkdir -p %s" % dir]
    copy_cmd = ["scp", engine_path, "picocluster@%s:%s" % (node.ip_address, dir)]

    subprocess.run(mkdir_cmd)
    subprocess.run(copy_cmd)


class Node(object):
    """ Represent a node in cluster """

    def __init__(self, node: client.models.v1_node.V1Node):
        """
        Retrieve node information from K8s
        Args:
            node: node object get from K8s
        """
        # Node name
        self.node_name = node.metadata.labels["kubernetes.io/hostname"]

        # IP address
        self.ip_address = node.metadata.annotations["flannel.alpha.coreos.com/public-ip"]
        self.hostname = "picocluster"    # default hostname

        # Total memory
        self.total_memory = get_number(node.status.capacity["memory"])
        self.total_memory = self.total_memory / (1 << 10)   # convert to Mb

        self.frontend_services = {}
        self.backend_services = {}

    def free_memory(self):
        """ Allocatable memory """
        # label_selector = "kubernetes.io/hostname=%s" % self.node_name
        # node = api.list_node(label_selector=label_selector).items[0]
        # free_mem = get_number(node.status.allocatable["memory"])

        cmd = ["ssh", "%s@%s" % (self.hostname, self.ip_address),
               "free", "--mega", "|", "grep", "Mem", "|", "awk", "'{print $NF}'"]

        res = subprocess.run(cmd, stdout=subprocess.PIPE)

        if res.returncode == 0:
            mem = int(res.stdout)
        else:
            mem = 0

        return mem

    def get_service_memory_used(self):
        """ Return the total amount of memory used by services running on this node """
        mem_sum = 0

        for _, service in self.frontend_services.items():
            mem_sum += service.memory

        for _, service in self.backend_services.items():
            mem_sum += service.memory

        return mem_sum

    def __str__(self):
        """ Print node info """
        pattern = "{:25}: {}\n"
        res = "\n"
        res += pattern.format("Node name", self.node_name)
        res += pattern.format("Ip address", self.ip_address)
        res += pattern.format("Memory capacity", self.total_memory)

        if not self.frontend_services:
            res += pattern.format("Frontend services", "")
        else:
            service_names = list(self.frontend_services.keys())
            res += pattern.format("Frontend services", service_names[0])
            for i in range(1, len(service_names)):
                res += "{:26} {}\n".format("", service_names[i])

        if not self.backend_services:
            res += pattern.format("Backend services", "")
        else:
            service_names = list(self.backend_services.keys())
            res += pattern.format("Backend services", service_names[0])
            for i in range(1, len(service_names)):
                res += "{:26} {}\n".format("", service_names[i])

        return res


class Service(object):
    """ Abstract class for service """

    def __init__(self, service_name: str, node: Node, memory: int, docker_image: str, port: int=8080):
        """ Data structure initialization """
        self.service_name = service_name
        self.node = node
        self.memory = memory
        self.docker_image = docker_image
        self.port = port
        self.exposed_service = None
        self.dev_type = ""

    def get_k8s_deployment_object(self) -> client.V1Deployment:
        """
        Create kubernetes deployment object
        Args:

        Returns:
            deployment: deployment object

        """
        if not self.node:
            raise ValueError("Target node is not specified")

        # Create container
        container = client.V1Container(name=self.service_name)
        container.image = self.docker_image
        container.image_pull_policy = "IfNotPresent"

        port = client.V1ContainerPort(container_port=self.port)

        container.ports = [port]

        # Create Pod spec
        pod_spec = client.V1PodSpec(containers=[container])
        pod_spec.node_selector = {"kubernetes.io/hostname": self.node.node_name}

        # Create Pod metadata
        pod_metadata = client.V1ObjectMeta()
        pod_metadata.labels = dict(app=self.service_name)

        # Create depolyment template
        template = client.V1PodTemplateSpec()
        template.metadata = pod_metadata
        template.spec = pod_spec

        # Create deployment spec
        selector = client.V1LabelSelector()
        selector.match_labels = dict(app=self.service_name)

        deployment_spec = client.V1DeploymentSpec(selector=selector, template=template)
        deployment_spec.replicas = 1  # explicitly set replicas to 1

        # Create metadata
        deployment_metadata = client.V1ObjectMeta()
        deployment_metadata.name = self.service_name
        deployment_metadata.labels = dict(app=self.service_name)

        # Put them all together to create Deployment object
        deployment = client.V1Deployment()
        deployment.api_version = "apps/v1"
        deployment.kind = "Deployment"
        deployment.metadata = deployment_metadata
        deployment.spec = deployment_spec

        return deployment

    def get_k8s_service_object(self) -> client.V1Service:
        """
        Return K8s service object
        Returns:
            k8s_service:
        """

        # Create port
        ports = [client.V1ServicePort(port=self.port, protocol="TCP", target_port=self.port)]

        # Create selector
        selector = dict(app=self.service_name)

        # Create service spec
        service_spec = client.V1ServiceSpec(selector=selector, ports=ports)

        # Create metadata
        service_metadata = client.V1ObjectMeta(name=self.service_name)

        # Create service object
        k8s_service = client.V1Service(api_version="v1", kind="Service", metadata=service_metadata, spec=service_spec)

        return k8s_service


def k8s_deploy(apps_v1_api: client.AppsV1Api, v1_api: client.CoreV1Api, service: Service):
    """ Deploy using K8s API and print info """
    print("[INFO] Deploying %s to %s..." % (service.service_name, service.node.node_name))
    apps_v1_api.create_namespaced_deployment("default", service.get_k8s_deployment_object())

    print("[INFO] Exposing %s " % service.service_name)
    service.exposed_service = v1_api.create_namespaced_service("default", service.get_k8s_service_object())


def k8s_update_deployment(apps_v1_api: client.AppsV1Api, service: Service):
    """ Update a Kubernetes deployment """
    print("[INFO] Updating deployment %s " % service.service_name)
    apps_v1_api.patch_namespaced_deployment(service.service_name, "default", service.get_k8s_deployment_object())

