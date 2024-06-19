[![Unit Tests and Style Checks](https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml/badge.svg)](https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml)
[![Build and Deploy Sphinx Documentation](https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml/badge.svg)](https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml)
[![Upload Python Package](https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml/badge.svg)](https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml)

<p align="center">
  <img src="https://drive.switch.ch/index.php/s/DAaKZEh9OeuvTEQ/download" alt="AdvSecureNet" width="400" />
</p>

AdvSecureNet is a Python library to for Machine Learning Security. It has been developed by [Melih Catal](https://github.com/melihcatal) at [University of Zurich](https://www.uzh.ch/en.html) as a part of his Master's Thesis under the supervision of [Prof. Dr. Manuel Günther](https://www.ifi.uzh.ch/en/aiml/people/guenther.html). Currently, the main focus of the library is on adversarial attacks and defenses on vision tasks. However, it's planned to extend the library to support other tasks such as natural language processing.

The library provides a set of tools to generate adversarial examples and to evaluate the robustness of machine learning models against adversarial attacks. It also provides a set of tools to train robust machine learning models. The library is built on top of [PyTorch](https://pytorch.org/). It is designed to be modular and extensible. So, anyone can easily run experiments with different configurations.

The library currently supports the following attacks:

- [FGSM - FGSM Targeted](https://arxiv.org/abs/1412.6572)
- [PGD - PGD Targeted](https://arxiv.org/abs/1706.06083)
- [DeepFool](https://arxiv.org/abs/1511.04599)
- [CW](https://arxiv.org/abs/1608.04644)
- [LOTS](https://arxiv.org/abs/1611.06179)
- [Decision Boundary](https://arxiv.org/abs/1712.04248)

The library currently supports the following defenses:

- [Adversarial Training](https://arxiv.org/abs/1412.6572)
- [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204)

The library supports any model that is implemented in PyTorch. It also provides a set of pre-trained models that can be used for experiments. It's also possible to create and use custom models.

The library supports multi-GPU training and adversarial training with DDP (Distributed Data Parallel) from PyTorch. This allows the library to be used for large-scale experiments.

## Installation

You can install the library using `pip`:

```bash
pip install advsecurenet
```

You can also install the library from source:

```bash
git clone
cd advsecurenet
pip install -e .
```

## Why AdvSecureNet?

- **Research-Oriented**: The library is designed to be research-oriented. It provides a set of tools to generate adversarial examples, evaluate the robustness of machine learning models against adversarial attacks, and train robust machine learning models. You can easily run experiments with different configurations using the yml configuration files. This helps you to run different experiments and compare the results with minimal effort. Additionally, you can easily share your experiments with others which helps to reproduce the results.

- **Supports Various Attacks and Defenses**: The library supports various adversarial attacks and defenses. You can easily run experiments with different attacks and defenses.

- **Supports Any PyTorch Model**: The library supports any model that is implemented in PyTorch. You can use pre-trained models or create custom models. You can also use your own models and datasets.

- **Supports Various Evaluation Metrics**: The library supports various evaluation metrics to evaluate the robustness of machine learning models against adversarial attacks, the performance of adversarial attacks, and the performance of adversarial defenses.

- **Native Multi-GPU Support**: Adversarial attacks, training, defenses, and evaluation can be computationally expensive. The library supports multi-GPU adversarial attacks, training, and evaluation. This allows you to run large-scale experiments with minimal effort, utilizing the computational resources efficiently.

## Usage

The library can be used as a command line tool or as an importable Python package.

### Command Line Tool

`advsecurenet` command can be used to interact with the library. You can use `advsecurenet --help` to see the available commands and options. Available commands are:

### Python Package

You can import the library as a Python package. You can use the `advsecurenet` module to access the library. You can find the available modules and classes in the [documentation](http://melihcatal.github.io/advsecurenet/).

## Examples

You can find various examples in the [examples](./examples/) directory. The examples show different use cases of the library and how to use the library as a Python package/CLI tool.

## Architecture

The high-level architecture of the library is shown in the figure below.

![advsecurenet_arch-2](https://github.com/melihcatal/advsecurenet/assets/46859098/cd3823b7-1402-4711-a1ab-e13b270de5d4)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE) for more details.

## Further Information

Further information about the library can be found in the [documentation](http://melihcatal.github.io/advsecurenet/).
