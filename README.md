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
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://drive.switch.ch/index.php/s/DAaKZEh9OeuvTEQ/download" alt="AdvSecureNet" width="400" />
</p>

AdvSecureNet is a Python library for Machine Learning Security, developed by [Melih Catal](https://github.com/melihcatal) at [University of Zurich](https://www.uzh.ch/en.html) as part of his Master’s Thesis under the supervision of [Prof. Dr. Manuel Günther](https://www.ifi.uzh.ch/en/aiml/people/guenther.html). The main focus of the library is on adversarial attacks and defenses for vision tasks, with plans to extend support to other tasks such as natural language processing.

The library provides tools to generate adversarial examples, evaluate the robustness of machine learning models against adversarial attacks, and train robust machine learning models.
Built on top of [PyTorch](https://pytorch.org/), it is designed to be modular and extensible, making it easy to run experiments with different configurations.
AdvSecureNet supports multi-GPU setups to enhance computational efficiency and fully supports both CLI and API interfaces, along with external YAML configuration files, enabling comprehensive testing and evaluation, facilitating the sharing and reproducibility of experiments.

## Table of Contents

- [Features](#features)
- [Supported Attacks](#supported-attacks)
- [Supported Defenses](#supported-defenses)
- [Supported Evaluation Metrics](#supported-evaluation-metrics)
- [Why AdvSecureNet?](#why-advsecurenet)
- [Comparison with Other Libraries](#comparison-with-other-libraries)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Tool](#command-line-tool)
  - [Python Package](#python-package)
- [Examples](#examples)
- [Architecture](#architecture)
- [Testing](#testing)
- [Quality Assurance](#quality-assurance)
- [License](#license)
- [Further Information](#further-information)

## Features

**Adversarial Attacks:** AdvSecureNet supports a diverse range of evasion attacks on computer vision tasks, including gradient-based, decision-based, single-step, iterative, white-box, black-box, targeted, and untargeted attacks, enabling comprehensive testing and evaluation of neural network robustness against various types of adversarial examples.

**Adversarial Defenses:** The toolkit includes adversarial training and ensemble adversarial training. Adversarial training incorporates adversarial examples into the training process to improve model robustness, while ensemble adversarial training uses multiple models or attacks for a more resilient defense strategy.

**Evaluation Metrics:** AdvSecureNet supports metrics like accuracy, robustness, transferability, and similarity. Accuracy measures performance on clean data, robustness assesses resistance to attacks, transferability evaluates how well adversarial examples deceive different models, and similarity quantifies perceptual differences using PSNR and SSIM.

**Multi-GPU Support:** AdvSecureNet is optimized for multi-GPU setups, enhancing the efficiency of training, evaluation, and adversarial attack generation, especially for large models and datasets or complex methods. By utilizing multiple GPUs in parallel, AdvSecureNet aims to reduce computational time, making it ideal for large-scale experiments and deep learning models.

**CLI and API Interfaces:** AdvSecureNet offers both CLI and API interfaces. The CLI allows for quick execution of attacks, defenses, and evaluations, while the API provides advanced integration and extension within user applications.

**External Configuration Files:** The toolkit supports YAML configuration files for easy parameter tuning and experimentation. This feature enables users to share experiments, reproduce results, and manage setups effectively, facilitating collaboration and comparison.

**Built-in Models and Datasets Support:** AdvSecureNet supports all PyTorch vision library models and well-known datasets like CIFAR-10, CIFAR-100, MNIST, FashionMNIST, and SVHN. Users can start without additional setup, but the toolkit also allows for custom datasets and models, offering flexibility for various research and applications.

**Automated Adversarial Target Generation:** AdvSecureNet can automatically generate adversarial targets for targeted attacks, simplifying the process and ensuring consistent and reliable results. As a user, you don't need to manually specify targets. This feature is especially useful for targeted attacks on large datasets. You can also provide custom targets if you prefer.

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

## Supported Evaluation Metrics

- Benign Accuracy
- Attack Success Rate
- Transferability
- Perturbation Distance
- Robustness Gap
- Perturbation Effectiveness

### Similarity Metrics

- [PSNR - Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [SSIM - Structural Similarity Index](https://en.wikipedia.org/wiki/Structural_similarity)

## Why AdvSecureNet?

- **Research-Oriented**: Easily run and share experiments with different configurations using YAML configuration files.

- **Supports Various Attacks and Defenses**: Experiment with a wide range of adversarial attacks and defenses.

- **Supports Any PyTorch Model**: Use pre-trained models or your own PyTorch models with the library.

- **Supports Various Evaluation Metrics**: Evaluate the robustness of models, performance of adversarial attacks, and defenses.

- **Bening Use Case Support**: Train and evaluate models on benign data.

- **Native Multi-GPU Support**: Efficiently run large-scale experiments utilizing multiple GPUs.

- **CLI and API Support**: Use the command line interface for quick experiments or the Python API for advanced integration.

- **Automated Adversarial Target Generation**: Simplify targeted attacks by letting the library generate targets automatically.

- **Active Maintenance**: Regular updates and improvements to ensure the library remains relevant and useful.

- **Comprehensive Documentation**: Detailed documentation to help you get started and make the most of the library.

- **Open Source**: Free and open-source under the MIT license, allowing you to use, modify, and distribute the library.

## Comparison with Other Libraries

AdvSecureNet stands out among adversarial machine learning toolkits like IBM ART, AdverTorch, SecML, FoolBox, ARES, and CleverHans. Key advantages include:

• **Active Maintenance:** Ensures ongoing support and updates.
• **Comprehensive Training Support:** One of the few toolkits supporting both adversarial and ensemble adversarial training.
• **Multi-GPU Support:** The first toolkit with native multi-GPU support for attacks, defenses, and evaluations, ideal for large-scale experiments.
• **Flexible Interfaces:** The first toolkit that fully supports CLI, API usage, and external YAML configuration files for reproducibility for all features.
• **Performance:** AdvSecureNet excels in performance, significantly reducing execution times on multi-GPU setups. For example, the multi-GPU PGD attack time (107 seconds) is faster than ARES’s best single GPU time (183 seconds). Adversarial training time is reduced from 304 seconds on a single GPU to 166 seconds with 7 GPUs, a speedup of 1.83x.

<img width="592" alt="image" src="https://github.com/melihcatal/advsecurenet/assets/46859098/e88658d3-1852-43ed-8c37-530e7fb7a0ce">

<img width="586" alt="image" src="https://github.com/melihcatal/advsecurenet/assets/46859098/0e0498dc-d3ea-41c4-a634-b4df5bbb0ce1">

[1] SecML supports attacks from CleverHans and FoolBox
[2] This feature is only available for adversarial training.

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

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/f3f86817-8ac3-4523-8f5e-cc9d4b4cbcf3)
_Usage example of AdvSecureNet demonstrating the equivalence between a YAML configuration file with a command-line interface (CLI) command and a corresponding Python API implementation._

## Examples

Examples of different use cases can be found in the [examples](./examples/) directory.

## Architecture

The high-level architecture of the toolkit is shown in the figure below.

![advsecurenet_arch](https://drive.switch.ch/index.php/s/SdKAyOZs1d9bcin/download)

![cli-arch](https://drive.switch.ch/index.php/s/ZbjIHBHql0dV6n0/download)

The toolkit is designed to be modular and extensible. CLI and Python API are implemented separately, however, they share the same core components and they have the same package structure for the sake of consistency. Tests are implemented for both CLI and Python API to ensure the correctness of the implementation and again they follow the same structure. The toolkit is designed to be easily extensible, new attacks, defenses, and evaluation metrics can be added by implementing the corresponding classes and registering them in the corresponding registries.

## Testing

The library is tested using `pytest` and coverage is measured using `coverage`. You can run the tests using the following command:

```bash
pytest tests/
```

Some tests take longer to run. To speed up the tests, you can use the `--device` option to run tests on a specific device (e.g., `--device cuda:0`).

```bash
pytest tests/ --device cuda:0
```

Tests are categorized into the following groups:

- `cli:` tests for the command line interface
- `advsecurenet:` tests for the Python API
- `essential:` tests for essential functionality (e.g., smoke and unit tests)
- `comprehensive:` tests for comprehensive functionality (e.g., integration tests)
- `extended:` tests for extended functionality (e.g., performance tests, security tests)

You can run tests for a specific group using the `m` option and the group name. For example, to run tests for the CLI:

```bash
pytest tests/ -m cli
```

CI/CD pipelines are set up to run tests automatically on every push and pull request. You can see the status of the tests in the badges at the top of the README.

## Quality Assurance

AdvSecureNet is designed with a strong emphasis on code quality and maintainability. The toolkit follows best practices in software engineering and ensures high standards through the following measures:

- **PEP 8 Compliance**: The codebase adheres to PEP 8 guidelines, the de facto coding standard for Python. We use [Black](https://github.com/psf/black) for automatic code formatting to maintain consistent style and readability.

- **Static Code Analysis**: We employ [Pylint](https://www.pylint.org/) for static code analysis and [MyPy](http://mypy-lang.org/) for type checking. These tools help catch potential errors and enforce coding standards.

- **Code Quality and Complexity**: Tools like [SonarQube](https://www.sonarqube.org/) and [Radon](https://radon.readthedocs.io/) provide insights into code quality and complexity. These tools are integrated into our CI/CD pipelines to ensure that the code remains clean and maintainable.

- **Comprehensive Testing**: The project features a robust testing suite, ensuring that all components are thoroughly tested. This helps in maintaining the reliability and stability of the toolkit.

- **Continuous Integration/Continuous Deployment (CI/CD)**: CI/CD pipelines are set up to automate the testing, analysis, and deployment processes. This ensures that any changes to the codebase are automatically verified for quality and correctness before being merged.

- **Documentation**: Comprehensive documentation is available on [GitHub Pages](https://melihcatal.github.io/advsecurenet/), providing detailed guidance on installation, usage, and API references. This ensures that users and contributors can easily understand and work with the toolkit.

By adhering to these practices and leveraging these tools, AdvSecureNet maintains a high standard of code quality, ensuring a reliable and user-friendly experience for developers and researchers alike.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE) for more details.

## Further Information

More information about the library can be found in the [documentation](http://melihcatal.github.io/advsecurenet/) and in the paper when it is published (hopefully soon :smile:).
