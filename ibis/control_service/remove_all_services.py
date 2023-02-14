#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 9/26/20
# Description: 
# ========================================================

from kubernetes import client, config

config.load_kube_config()

k8s_core_v1 = client.CoreV1Api()
k8s_services = k8s_core_v1.list_namespaced_service(namespace="default").items

for service in k8s_services:
    if service.metadata.name != "kubernetes":
        print("[INFO] Deleted service %s" % service.metadata.name)
        k8s_core_v1.delete_namespaced_service(service.metadata.name, namespace="default")

