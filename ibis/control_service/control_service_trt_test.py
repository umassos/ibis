#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/5/20
# Description: 
# ========================================================

import os
import time
import control_service_trt
import control_service_base


def check_nodes(control: control_service_trt.ControlServiceGPU):
    """ Check nodes in control services """
    res = []
    for _, node in control.nodes.items():
        res.append("=========================================================================")
        res.append(node.__str__())
        res.append("Available memory: %d" % node.free_memory())

    res = "\n".join(res)

    print(res, end="\r")


def check_deploy(control: control_service_trt.ControlServiceGPU):
    """ Deploy a simple job """
    frontend_image = "washraf/frontend-nano"
    backend_image = "washraf/backend-nano"
    model_path = os.path.join(os.getcwd(), "models/extra")
    model_path = os.path.join(model_path, "ResNet101.onnx")
    model_config = control_service_trt.ModelConfigGPU(model_name="test-task",
                                                      model_path=model_path,
                                                      model_type="classification",
                                                      build_engine=False,
                                                      input_rate=10, service_rate=10,
                                                      frontend_memory=100,
                                                      backend_memory=200,
                                                      frontend_image=frontend_image,
                                                      backend_image=backend_image, sla=30,
                                                      input_shape="1x3x224x224",
                                                      wait_time=0,
                                                      output_bindings=["output"],
                                                      dynamic_shapes=("input:1x3x224x224",
                                                                      "input:32x3x224x224",
                                                                      "input:8x3x224x224"))

    control.deploy(model_config)


def check_copy_file(control: control_service_trt.ControlServiceGPU):
    filepath = os.path.join(os.getcwd(), "control_service_edgetpu_test.py")
    control_service_base.copy_to_host(filepath, control.nodes["jetson4"])


def main():

    # Check nodes
    control = control_service_trt.ControlServiceGPU()

    # Deploy a simple task
    check_deploy(control)

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
