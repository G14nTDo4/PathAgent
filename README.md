# PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/G14nTDo4/PathAgent)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17052-b31b1b.svg)](https://arxiv.org/abs/2511.17052)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2511.17052.pdf)

<div align="center">

</div>

Official code implementation for paper "PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning"

>Jingyun Chen, Linghan Cai, Zhikang Wang, Yi Huang, Songhan Jiang, Shenjin Huang, Hongpeng Wang, Yongbing Zhang


## Overview

PathAgent is the first training-free interactive agent specifically designed for WSI analysis. By coordinating off-the-shelf pathology models through an agent, it yields traceable decisions and competitive accuracy, suggesting a pragmatic route of computational pathology.

The contributions of PathAgent can be summarized in three aspects:

1. Dynamic analytic Logic: We replace single-step reasoning with Multi-Step Reasoning in the Executor. This mechanism can construct analytic logic and dynamically provide guidelines to retrieve task-relevant information.
2. Adaptive Magnification: PathAgent can adaptively select an appropriate scale based on the analytic state, generating more refined visual evidence.
3. Enhanced Evidence Retrieval: We improve the accuracy of evidence capture by simplifying the query strategy of the Navigator.


![architecture](./assets/Overview.png)
<p align="center"><i>Overview of PathAgent</i></p>

![architecture](./assets/CaseStudy.png)
<p align="center"><i>Illustration of PathAgent's inference procedure</i></p>

## Citation
```bibtex
@article{chen2025pathagent,
      title={PathAgent: Toward Interpretable Analysis of Whole-slide Pathology Images via Large Language Model-based Agentic Reasoning}, 
      author={Jingyun Chen and Linghan Cai and Zhikang Wang and Yi Huang and Songhan Jiang and Shenjin Huang and Hongpeng Wang and Yongbing Zhang},
      journal={arXiv preprint arXiv:2511.17052},
      year={2025}
}
```
