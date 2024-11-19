---
title: "AdvSecureNet: A Python Toolkit for Adversarial Machine Learning"
tags:
  - Python
  - Machine Learning
  - Adversarial Machine Learning
  - Trustworthy Machine Learning
  - PyTorch
authors:
  - name: Melih Catal
    orcid: 0009-0009-0231-287X
    affiliation: 1
    corresponding: true
  - name: Manuel Günther
    orcid: 0000-0003-1489-7448
    affiliation: 2
affiliations:
  - name: Software Evaluation and Architecture Lab, University of Zurich, Zurich, Switzerland
    index: 1
  - name: Artificial Intelligence and Machine Learning Group, University of Zurich, Zurich, Switzerland
    index: 2
date: 19 November 2024
bibliography: paper.bib
---

# Summary

Machine learning models are vulnerable to adversarial attacks. Several tools have been developed to research these vulnerabilities, but they often lack comprehensive features and flexibility.

We introduce **AdvSecureNet**, a PyTorch-based toolkit for adversarial machine learning that is the first to natively support multi-GPU setups for attacks, defenses, and evaluation. It is the first toolkit that supports both CLI and API interfaces and external YAML configuration files to enhance versatility and reproducibility.

The toolkit includes multiple attacks, defenses, and evaluation metrics. Rigorous software engineering practices are followed to ensure high code quality and maintainability.

The project is available as an open-source project on GitHub at [https://github.com/melihcatal/advsecurenet](https://github.com/melihcatal/advsecurenet) and installable via PyPI.

# Statement of Need

Machine learning models are increasingly deployed in critical applications[@biswas2023role; @mintz2019introduction], including autonomous vehicles [@bojarski2016end], facial recognition systems [@parmar2014face; @guenther2016survey], and natural language processing [@patwardhan2023transformers]. However, these models are highly vulnerable to adversarial attacks—subtle perturbations to input data that can mislead models into making incorrect predictions [@fgsm_goodfellow; @Szegedy2013IntriguingPO]. This poses a significant challenge to the reliability and security of machine learning systems [@khalid2021exploiting_attacks].

Several libraries, such as ART [@art2018], AdverTorch [@ding2019advertorch], and CleverHans [@cleverhans], provide tools for implementing and testing adversarial attacks and defenses. However, they suffer from key limitations, including:

- Lack of multi-GPU support for large-scale experiments.
- Limited configurability and flexibility.
- Lack of robust interfaces for reproducible research and experimentation.

AdvSecureNet, addresses these gaps by offering:

- **Multi-GPU Support:** Optimized for adversarial training, attack generation, and evaluation, making it suitable for large-scale experiments.
- **Versatile Interfaces:** Provides both CLI and API interfaces for diverse use cases and flexible integration with existing workflows.
- **Configurability:** Supports YAML-based configuration files for easy experiment sharing, parameter tuning, and reproducibility.
- **Comprehensive Features:** Includes a range of adversarial attacks (e.g., gradient-based, decision-based, targeted, and untargeted attacks) and defenses (e.g., adversarial training and ensemble adversarial training) [@fgsm_goodfellow; @Tramr2017EnsembleAT].
- **Evaluation Metrics:** Supports metrics like accuracy, robustness, transferability, and perceptual similarity using PSNR [@hore2010image] and SSIM [@Wang2004ImageQA].
- **Prebuilt Models and Datasets:** Seamless integration with datasets (CIFAR-10, CIFAR-100, ImageNet, etc.) and PyTorch models, enabling out-of-the-box experimentation.

The following tables highlight how AdvSecureNet compares to other libraries in terms of features and performance.

**Feature Comparison**

| Feature                        | AdvSecureNet | IBM Art | AdverTorch | SecML[@melis2019secml] | FoolBox[@rauber2017foolbox] | Ares [@dong2020benchmarkingares] | CleverHans |
| ------------------------------ | ------------ | ------- | ---------- | ---------------------- | --------------------------- | -------------------------------- | ---------- |
| **Actively Maintained**        | Yes          | Yes     | No         | No                     | No                          | No                               | No         |
| **Last Year of Contribution**  | 2024         | 2024    | 2022       | 2024                   | 2024                        | 2023                             | 2023       |
| **PyTorch Support**            | Yes          | Yes     | Yes        | Yes                    | Yes                         | Yes                              | Yes        |
| **TensorFlow Support**         | No           | Yes     | No         | Yes                    | Yes                         | No                               | Yes        |
| **Number of Attacks**          | 8            | 60      | 17         | 39                     | 31                          | 28                               | 8          |
| **Number of Defenses**         | 2            | 37      | 3          | -                      | -                           | 3                                | 1          |
| **Evaluation Metrics**         | 6            | 5       | -          | -                      | 2                           | 1                                | 2          |
| **Built-in Multi-GPU Support** | Yes          | No      | No         | No                     | No                          | Limited                          | No         |
| **API Usage**                  | Yes          | Yes     | Yes        | Yes                    | Yes                         | Yes                              | Yes        |
| **CLI Usage**                  | Yes          | No      | No         | No                     | No                          | Limited                          | No         |
| **External Config File**       | Yes          | No      | No         | No                     | No                          | Limited                          | No         |

The following table highlights the performance of AdvSecureNet compared to other toolkits:

**Performance Benchmark for AdvSecureNet and Other Toolkits**

| Metric                               | Toolkit      | Dataset  | Single GPU Time (min) | Multi-GPU Time (min)             | Speedup                        |
| ------------------------------------ | ------------ | -------- | --------------------- | -------------------------------- | ------------------------------ |
| **FGSM Attack**                      | AdvSecureNet | CIFAR-10 | 0.4                   | 0.37 (4 GPUs), **0.24 (7 GPUs)** | 1.09x (4 GPUs), 1.64x (7 GPUs) |
|                                      | IBM ART      | CIFAR-10 | 0.82                  | N/A                              | N/A                            |
|                                      | CleverHans   | CIFAR-10 | 0.25                  | N/A                              | N/A                            |
|                                      | ARES         | CIFAR-10 | 0.45                  | N/A                              | N/A                            |
|                                      | FoolBox      | CIFAR-10 | 0.38                  | N/A                              | N/A                            |
|                                      | AdverTorch   | CIFAR-10 | **0.19**              | N/A                              | N/A                            |
| **PGD-20 Attack**                    | AdvSecureNet | CIFAR-10 | 3.48                  | 2.47 (4 GPUs), **1.78 (7 GPUs)** | 1.41x (4 GPUs), 1.95x (7 GPUs) |
|                                      | IBM ART      | CIFAR-10 | 11.0                  | N/A                              | N/A                            |
|                                      | CleverHans   | CIFAR-10 | 3.87                  | N/A                              | N/A                            |
|                                      | ARES         | CIFAR-10 | **3.05**              | N/A                              | N/A                            |
|                                      | FoolBox      | CIFAR-10 | 3.67                  | N/A                              | N/A                            |
|                                      | AdverTorch   | CIFAR-10 | 3.63                  | N/A                              | N/A                            |
| **Adversarial Training on CIFAR-10** | AdvSecureNet | CIFAR-10 | 5.07                  | 4.03 (4 GPUs), **2.77 (7 GPUs)** | 1.26x (4 GPUs), 1.83x (7 GPUs) |
|                                      | ARES         | CIFAR-10 | 15.9                  | 12.0 (4 GPUs), 12.8 (7 GPUs)     | 1.33x (4 GPUs), 1.24x (7 GPUs) |
|                                      | IBM ART      | CIFAR-10 | **4.87**              | N/A                              | N/A                            |
| **Adversarial Training on ImageNet** | AdvSecureNet | ImageNet | **240**               | 33 (4 GPUs), **30 (7 GPUs)**     | 7.27x (4 GPUs), 8x (7 GPUs)    |
|                                      | ARES         | ImageNet | 627                   | 313 (4 GPUs), 217 (7 GPUs)       | 2.0x (4 GPUs), 2.89x (7 GPUs)  |
|                                      | IBM ART      | ImageNet | 323                   | N/A                              | N/A                            |

# References
