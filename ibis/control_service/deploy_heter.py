#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 1/5/21
# Description: 
# ========================================================

import argparse
import yaml
import time
import os
import logging
import numpy as np

from control_service_heter import ModelConfig, ControlService
from typing import List

LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)


def load_model_config(filename: str) -> List[ModelConfig]:
    """
    Load workloads specified in YAML file
    Args:
        filename: path to the YAML file
        model_type: default model type

    Returns:
        model_configs: list of loaded model configurations

    """
    with open(filename, 'r') as f:
        data = yaml.load(f.read())

    model_configs = []
    for model in data:
        model_config = ModelConfig(**model)
        model_configs.append(model_config)
    return model_configs


def check_nodes(control: ControlService):
    """ Check nodes in control services """
    res = []
    for _, node in control.nodes.items():
        res.append("=========================================================================")
        res.append(node.__str__())
        res.append("Available memory: %d" % node.free_memory())

    res = "\n".join(res)

    print(res, end="\r")


def main():
    parser = argparse.ArgumentParser(description="Deploy AI models on GPU and TPU cluster")
    parser.add_argument("-f", "--workload-file", required=True, type=str, dest="workload_filename",
                        help="Path to the .yaml workload file")
    parser.add_argument("--gpu-method", type=str, dest="gpu_method", default="modeling_bfs", help="gpu method")
    parser.add_argument("--tpu-method", type=str, dest="tpu_method", default="bfs", help="tpu method")
    parser.add_argument("--dev-method", type=str, dest="dev_method", default="best", help="method for choosing device")
    parser.add_argument("-n", "--num-models", type=int, dest="num_models", default=1000,
                        help="The maximum number of models to place")
    parser.add_argument("-t", "--workload-type", type=str, default="MAAS", dest="workload_type",
                        help="Workload type, MAAS or user_trained")
    args = parser.parse_args()

    logger = logging.getLogger("Deploy models")
    logger.setLevel(logging.INFO)

    np.random.seed(404)

    log_name = os.path.basename(args.workload_filename)
    log_name = os.path.splitext(log_name)[0] + "_" + args.workload_type + "_controller.csv"
    log_name = os.path.join("results/", log_name)

    model_configs = load_model_config(args.workload_filename)
    control_service = ControlService(tpu_method=args.tpu_method, gpu_method=args.gpu_method, dev_method=args.dev_method)

    running_models = []
    running_model_configs = []
    backend_models = {}

    deploy_cnt = 0
    for i, config in enumerate(model_configs):
        if deploy_cnt >= args.num_models:
            break

        frontend_service, backend_service = control_service.deploy(config)
        if frontend_service and backend_service:
            logger.info("Successfully deploying %d model %s to the cluster..." % (i + 1, config.model_name))
            deploy_cnt += 1
            running_models.append({
                "frontend_service_name": frontend_service.service_name,
                "input_rate": frontend_service.input_rate,
                "dev_type": backend_service.dev_type
            })
            running_model_configs.append(dict(config._asdict()))
            if backend_service.service_name not in backend_models:
                logger.info("New Model was Added")
                backend_models[backend_service.service_name] = backend_service.service_name
        else:
            logger.warning("Fail to deploy %d model to the cluster, stop deploying further models" % (i+1))

    run_filename = os.path.join("results/", "current_running_models.yaml")

    total_input_rate = np.sum([s["input_rate"] for s in running_models])
    logger.info("Number of clients deployed: {},number of models {}, total input rate {}".format(len(running_models),
                                                                                                 len(backend_models),
                                                                                                 total_input_rate))
    logger.info("Saving running models to {}".format(run_filename))

    with open("workloads/running_workload.yaml", 'w') as f:
        yaml.dump(running_model_configs, f)
    with open(run_filename, 'w') as f:
        yaml.dump(running_models, f)

    with open("results/current_running_config", 'w') as f:
        f.write(args.workload_type + "_" + os.path.basename(args.workload_filename).split(".")[0])
    try:
        while True:
            # Check nodes
            check_nodes(control_service)

            time.sleep(10)
    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        control_service.destroy(log_name)


if __name__ == '__main__':
    main()
