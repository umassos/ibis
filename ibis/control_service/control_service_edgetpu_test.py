#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/5/20
# Description: 
# ========================================================

import os
import time
import control_service_edgetpu


def check_nodes(control: control_service_edgetpu.ControlServiceTPU):
    """ Check nodes in control services """
    res = []
    for _, node in control.nodes.items():
        res.append("=========================================================================")
        res.append(node.__str__())
        res.append("Available memory: %d" % node.free_memory())

    res = "\n".join(res)

    print(res, end="\r")


def check_deploy_tpu(control: control_service_edgetpu.ControlServiceTPU):
    """ Deploy a simple job """
    frontend_image = "qianlinliang/frontend-edgetpu-pi"
    backend_image = "qianlinliang/backend-edgetpu-pi"
    model_dir = "/home/picocluster/workspace/edgetpu/models/tflite"
    model_filenames = ["inception_v4.tflite", "mobilenet_v2.tflite"]
    model_types = ["classification", "classification"]
    input_shapes = ["1x299x299x3", "1x224x224x3"]
    input_rate = [2.764, 16.62]
    service_rate = [7.992, 203.67]
    switch_overhead = [19.43, 13.14]

    for i in range(len(model_filenames)):
        model_config = control_service_edgetpu.ModelConfigTPU(model_dir=model_dir, model_filename=model_filenames[i],
                                                              model_type=model_types[i],
                                                              input_rate=input_rate[i], service_rate=service_rate[i],
                                                              switch_overhead=switch_overhead[i],
                                                              frontend_memory=100,
                                                              backend_memory=200,
                                                              frontend_image=frontend_image,
                                                              backend_image=backend_image, sla=1000,
                                                              input_shape=input_shapes[i])

        control.deploy(model_config)


def main():

    # Check nodes
    control = control_service_edgetpu.ControlServiceTPU()

    # Deploy a simple task
    check_deploy_tpu(control)

    try:
        while True:
            # Check nodes
            check_nodes(control)

            time.sleep(5)
    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        control.destroy()


if __name__ == "__main__":
    main()