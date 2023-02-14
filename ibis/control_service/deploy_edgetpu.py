#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 9/11/20
# Description: 
# ========================================================

import argparse
import yaml
import time
import os
import logging
import numpy as np

from control_service_edgetpu import ModelConfigTPU, ControlServiceTPU
from typing import List


LOG_FORMAT = "[%(levelname)s] %(asctime)-15s %(message)s"
logging.basicConfig(format=LOG_FORMAT)


def load_model_config(filename: str) -> List[ModelConfigTPU]:
    """
    Load workloads specified in YAML file
    Args:
        filename: path to the YAML file

    Returns:
        model_configs: list of loaded model configurations

    """
    with open(filename, 'r') as f:
        data = yaml.load(f.read())

    model_configs = []
    for model in data["models"]:
        model_configs.append(ModelConfigTPU(model_dir=data["model_dir"], **model))

    return model_configs


def check_nodes(control: ControlServiceTPU):
    """ Check nodes in control services """
    res = []
    for _, node in control.nodes.items():
        res.append("=========================================================================")
        res.append(node.__str__())
        res.append("Available memory: %d" % node.free_memory())

    res = "\n".join(res)

    print(res, end="\r")


def main():
    parser = argparse.ArgumentParser(description="Deploy AI models on edgeTPU")
    parser.add_argument("-f", "--workload-file", required=True, type=str, dest="workload_filename",
                        help="Path to the .yaml workload file")
    parser.add_argument("-m", "--method", type=str, dest="method", default="bfs", help="placement method")
    parser.add_argument("-n", "--num-models", type=int, dest="num_models", default=1000,
                        help="The maximum number of models to place")
    args = parser.parse_args()

    logger = logging.getLogger("Deploy models")
    logger.setLevel(logging.INFO)

    np.random.seed(404)

    log_name = os.path.basename(args.workload_filename)
    log_name = os.path.splitext(log_name)[0] + "_controller.csv"
    log_name = os.path.join("results/", log_name)

    model_configs = load_model_config(args.workload_filename)

    control_service = ControlServiceTPU(method=args.method)

    running_models = []
    for i, config in enumerate(model_configs):
        if i >= args.num_models:
            break

        frontend_service, backend_service = control_service.deploy(config)
        if frontend_service and backend_service:
            logger.info("Successfully deploying %d model %s to the cluster..." % (i + 1, config.model_filename))
            running_models.append({
                "frontend_service_name": frontend_service.service_name,
                "input_rate": frontend_service.input_rate
            })
        else:
            logger.warning("Fail to deploy %d model to the cluster, stop deploying further models" % (i+1))

    run_filename = os.path.join("results/", "current_running_models.yaml")

    total_input_rate = np.sum([s["input_rate"] for s in running_models])
    logger.info("Number of models deployed: {}, total input rate {}".format(len(running_models), total_input_rate))
    logger.info("Saving running models to {}".format(run_filename))
    with open(run_filename, 'w') as f:
        yaml.dump(running_models, f)

    try:
        while True:
            time.sleep(10)
            # Check nodes
            control_service.detect_hotspot()
            check_nodes(control_service)

    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        control_service.destroy(log_name)


if __name__ == '__main__':
    main()
