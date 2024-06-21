# AdvSecureNet

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=bugs)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=coverage)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=sqale_index)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet)
[![Unit Tests and Style Checks](https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml/badge.svg?branch=develop)](https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml)
[![Build and Deploy Sphinx Documentation](https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml/badge.svg)](https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml)
[![Upload Python Package](https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml/badge.svg)](https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml)

<p align="center">
  <img src="https://drive.switch.ch/index.php/s/DAaKZEh9OeuvTEQ/download" alt="AdvSecureNet" width="400" />
</p>

AdvSecureNet is a Python library for Machine Learning Security, developed by [Melih Catal](https://github.com/melihcatal) at [University of Zurich](https://www.uzh.ch/en.html) as part of his Master's Thesis under the supervision of [Prof. Dr. Manuel Günther](https://www.ifi.uzh.ch/en/aiml/people/guenther.html). The main focus of the library is on adversarial attacks and defenses for vision tasks, with plans to extend support to other tasks such as natural language processing.

The library provides tools to generate adversarial examples, evaluate the robustness of machine learning models against adversarial attacks, and train robust machine learning models. Built on top of [PyTorch](https://pytorch.org/), it is designed to be modular and extensible, making it easy to run experiments with different configurations.

## Table of Contents

- [Features](#features)
- [Supported Attacks](#supported-attacks)
- [Supported Defenses](#supported-defenses)
- [Installation](#installation)
- [Why AdvSecureNet?](#why-advsecurenet)
- [Usage](#usage)
  - [Command Line Tool](#command-line-tool)
  - [Python Package](#python-package)
- [Examples](#examples)
- [Architecture](#architecture)
- [License](#license)
- [Further Information](#further-information)

## Features

- Generate adversarial examples
- Evaluate model robustness against adversarial attacks
- Train robust machine learning models
- Modular and extensible design
- Native multi-GPU support

## Supported Attacks

- [FGSM - FGSM Targeted](https://arxiv.org/abs/1412.6572)
- [PGD - PGD Targeted](https://arxiv.org/abs/1706.06083)
- [DeepFool](https://arxiv.org/abs/1511.04599)
- [CW](https://arxiv.org/abs/1608.04644)
- [LOTS](https://arxiv.org/abs/1611.06179)
- [Decision Boundary](https://arxiv.org/abs/1712.04248)

## Supported Defenses

- [Adversarial Training](https://arxiv.org/abs/1412.6572)
- [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204)

## Installation

You can install the library using `pip`:

```bash
pip install advsecurenet
```

Or install it from source:

```bash
git clone https://github.com/melihcatal/advsecurenet.git
cd advsecurenet
pip install -e .
```

## Why AdvSecureNet?

- **Research-Oriented**: Easily run and share experiments with different configurations using YAML configuration files.

- **Supports Various Attacks and Defenses**: Experiment with a wide range of adversarial attacks and defenses.

- **Supports Any PyTorch Model**: Use pre-trained models or your own PyTorch models with the library.

- **Supports Various Evaluation Metrics**: Evaluate the robustness of models, performance of adversarial attacks, and defenses.

- **Bening Use Case Support**: Train and evaluate models on benign data.

- **Native Multi-GPU Support**: Efficiently run large-scale experiments utilizing multiple GPUs.

## Usage

The library can be used as a command line tool or as an importable Python package.

### Command Line Tool

Use the `advsecurenet` command to interact with the library. Use `advsecurenet --help` to see available commands and options. It is recommended to use YAML configuration files to run experiments. You can list the available configuration options using `advsecurenet utils configs list` and generate a template configuration file using `advsecurenet utils configs get -c <config_name> -o <output_file>`.

Running an adversarial attack:

```bash
advsecurenet attack -c ./fgsm.yml
```

Running an adversarial defense:

```bash
advsecurenet defense adversarial-training -c ./adv_training.yml
```

Running an evaluation:

```bash
advsecurenet evaluate benign -c ./evaluate_benign.yml

or

advsecurenet evaluate adversarial -c ./evaluate_adversarial.yml
```

### Python Package

You can import the library as a Python package. You can use the `advsecurenet` module to access the library. You can find the available modules and classes in the [documentation](http://melihcatal.github.io/advsecurenet/).

## Examples

Examples of different use cases can be found in the [examples](./examples/) directory.

## Architecture

The high-level architecture of the library is shown in the figure below.

![advsecurenet_arch-2](https://github.com/melihcatal/advsecurenet/assets/46859098/cd3823b7-1402-4711-a1ab-e13b270de5d4)

## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE) for more details.

## Further Information

More information about the library can be found in the [documentation](http://melihcatal.github.io/advsecurenet/).
