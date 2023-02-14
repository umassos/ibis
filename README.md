## Ibis - Model-driven Cluster Resource Management for AI Workloads in Edge Clouds

Ibis is a queueing model-driven edge cloud resource management system that maximize edge accelerator sharing while meeting all applications' latency SLOs. 


This repository contains source code used in the paper: 

```bibtex
@inproceedings{taas23-ibis,
    author = {Qianlin Liang and Walid A. Hanafy and Ahmed Ali-Eldin and Prashant Shenoy},
    title = { Model-driven Cluster Resource Management for AI Workloads in Edge Clouds},
    month = 1,
    year = 2023,
    doi = {10.1145/3582080},
    booktitle = { ACM Transactions on Adaptive andAutonomous Systems (TAAS)},
}
```

## Contents

This repository has the following architectures: 

```
├── data
│   └── queueing_validation
├── ibis
│   ├── backend-service
│   ├── control_service
│   └── frontend-service
└── notebooks
```

The `ibis` folder includes source code of Ibis. 

* `frontend-service` contains source code for frontend containers, which perform the preprocessing for the inference service. Frontend containers are generally running on CPUs.
* `backend-service` contains source code for backend containers, which perform AI inference. Backend containers support running with accelerators, such as GPU and TPU. 
* `control_service` contains the source code for placing incoming workloads to the cluster, while respecting latency SLOs of the workloads.  

The `data` folder includes data we collect from experiments described in the paper. The `notebooks` folder includes code for ploting figures using experiment data. 



