#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 8/11/20
# Description: 
# ========================================================

from control_service_trt import placement_algorithms_trt


def build_engine_test():
    # Dynamic shape test
    print("Building engine with dynamic shapes")
    model_path = "models/resnet18.onnx"
    dynamic_shapes = ("input:1x3x224x224",
                      "input:32x3x224x224",
                      "input:8x3x224x224")
    workspace = 500

    placement_algorithms_trt.build_engine(model_path, dynamic_shapes, workspace)

    # Not dynamic shape test
    print("Building engine with not dynamic shape")
    model_path = "models/efficientnet-s.onnx"
    placement_algorithms_trt.build_engine(model_path, workspace=workspace)


def get_memory_size_test():
    engine_path1 = "models/yolov3.engine"
    engine_path2 = "models/efficientnet_b5.engine"

    memory1 = placement_algorithms_trt.get_trt_engine_memory_usage(engine_path1, max_batch_size=32)
    print("Memory usage for %s is: %d MB" % (engine_path1, memory1))
    memory2 = placement_algorithms_trt.get_trt_engine_memory_usage(engine_path2, max_batch_size=1)
    print("Memory usage for %s is: %d MB" % (engine_path2, memory2))


def profile_service_time_test():
    engine_path1 = "models/resnet18.engine"
    engine_path2 = "models/efficientnet-s.engine"

    service_time1 = placement_algorithms_trt.get_trt_service_time(engine_path1, input_shape="input:8x3x224x224")
    print("Service time for %s is: %.2f" % (engine_path1, service_time1))
    service_time2 = placement_algorithms_trt.get_trt_service_time(engine_path2)
    print("Service time for %s is: %.2f" % (engine_path2, service_time2))


def main():
    # Test build engine
    build_engine_test()

    # Test get memory size
    get_memory_size_test()

    # Test service time profile
    # profile_service_time_test()


if __name__ == "__main__":
    main()

