AdvSecureNet
============

|Quality Gate Status| |Bugs| |Code Smells| |Coverage| |Duplicated Lines
(%)| |Lines of Code| |Reliability Rating| |Security Rating| |Technical
Debt| |Maintainability Rating| |Vulnerabilities| |Unit Tests and Style
Checks| |Build and Deploy Sphinx Documentation| |Upload Python Package|

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

|advsecurenet_logo| 
AdvSecureNet is a Python library for Machine Learning Security, developed by `Melih Catal <https://github.com/melihcatal>`__ at `University of Zurich <https://www.uzh.ch/en.html>`__ as part of his Master’s Thesis under the supervision of `Prof. Dr. Manuel Günther <https://www.ifi.uzh.ch/en/aiml/people/guenther.html>`__. The main focus of the library is on adversarial attacks and defenses for vision tasks, with plans to extend support to other tasks such as natural language processing.

The library provides tools to generate adversarial examples, evaluate the robustness of machine learning models against adversarial attacks, and train robust machine learning models. 
Built on top of `PyTorch <https://pytorch.org/>`__, it is designed to be modular and extensible, making it easy to run experiments with different configurations. 
AdvSecureNet supports multi-GPU setups to enhance computational efficiency and fully supports both CLI and API interfaces, along with external YAML configuration files, enabling comprehensive testing and evaluation, facilitating the sharing and reproducibility of experiments.

Features
--------

**Adversarial Attacks:** AdvSecureNet supports a diverse range of
evasion attacks on computer vision tasks, including gradient-based,
decision-based, single-step, iterative, white-box, black-box, targeted,
and untargeted attacks, enabling comprehensive testing and evaluation of
neural network robustness against various types of adversarial examples.

**Adversarial Defenses:** The toolkit includes adversarial training and
ensemble adversarial training. Adversarial training incorporates
adversarial examples into the training process to improve model
robustness, while ensemble adversarial training uses multiple models or
attacks for a more resilient defense strategy.

**Evaluation Metrics:** AdvSecureNet supports metrics like accuracy,
robustness, transferability, and similarity. Accuracy measures
performance on clean data, robustness assesses resistance to attacks,
transferability evaluates how well adversarial examples deceive
different models, and similarity quantifies perceptual differences using
PSNR and SSIM.

**Multi-GPU Support:** AdvSecureNet is optimized for multi-GPU setups,
enhancing the efficiency of training, evaluation, and adversarial attack
generation, especially for large models and datasets or complex methods.
By utilizing multiple GPUs in parallel, AdvSecureNet aims to reduce
computational time, making it ideal for large-scale experiments and deep
learning models.

**CLI and API Interfaces:** AdvSecureNet offers both CLI and API
interfaces. The CLI allows for quick execution of attacks, defenses, and
evaluations, while the API provides advanced integration and extension
within user applications.

**External Configuration Files:** The toolkit supports YAML
configuration files for easy parameter tuning and experimentation. This
feature enables users to share experiments, reproduce results, and
manage setups effectively, facilitating collaboration and comparison.

**Built-in Models and Datasets Support:** AdvSecureNet supports all
PyTorch vision library models and well-known datasets like CIFAR-10,
CIFAR-100, MNIST, FashionMNIST, and SVHN. Users can start without
additional setup, but the toolkit also allows for custom datasets and
models, offering flexibility for various research and applications.

:doc:`Supported Attacks <attacks>`
-----------------

-  `FGSM - FGSM Targeted <https://arxiv.org/abs/1412.6572>`__
-  `PGD - PGD Targeted <https://arxiv.org/abs/1706.06083>`__
-  `DeepFool <https://arxiv.org/abs/1511.04599>`__
-  `CW <https://arxiv.org/abs/1608.04644>`__
-  `LOTS <https://arxiv.org/abs/1611.06179>`__
-  `Decision Boundary <https://arxiv.org/abs/1712.04248>`__

:doc:`Supported Defenses <defenses>`
------------------

-  `Adversarial Training <https://arxiv.org/abs/1412.6572>`__
-  `Ensemble Adversarial Training <https://arxiv.org/abs/1705.07204>`__

:doc:`Supported Evaluation Metrics <evaluations>`
----------------------------

-  Benign Accuracy
-  Attack Success Rate
-  Transferability
-  Perturbation Distance
-  Robustness Gap
-  Perturbation Effectiveness

Similarity Metrics
~~~~~~~~~~~~~~~~~~

-  `PSNR - Peak Signal-to-Noise
   Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__
-  `SSIM - Structural Similarity
   Index <https://en.wikipedia.org/wiki/Structural_similarity>`__

Installation
------------

You can install the library using ``pip``:

.. code:: bash

   pip install advsecurenet

Or install it from source:

.. code:: bash

   git clone https://github.com/melihcatal/advsecurenet.git
   cd advsecurenet
   pip install -e .

Why AdvSecureNet?
-----------------

-  **Research-Oriented**: Easily run and share experiments with
   different configurations using YAML configuration files.

-  **Supports Various Attacks and Defenses**: Experiment with a wide
   range of adversarial attacks and defenses.

-  **Supports Any PyTorch Model**: Use pre-trained models or your own
   PyTorch models with the library.

-  **Supports Various Evaluation Metrics**: Evaluate the robustness of
   models, performance of adversarial attacks, and defenses.

-  **Bening Use Case Support**: Train and evaluate models on benign
   data.

-  **Native Multi-GPU Support**: Efficiently run large-scale experiments
   utilizing multiple GPUs.

Usage
-----

The library can be used as a command line tool or as an importable
Python package.

Command Line Tool
~~~~~~~~~~~~~~~~~

Use the ``advsecurenet`` command to interact with the library. Use
``advsecurenet --help`` to see available commands and options. It is
recommended to use YAML configuration files to run experiments. You can
list the available configuration options using
``advsecurenet utils configs list`` and generate a template
configuration file using
``advsecurenet utils configs get -c <config_name> -o <output_file>``.

Running an adversarial attack:

.. code:: bash

   advsecurenet attack -c ./fgsm.yml

Running an adversarial defense:

.. code:: bash

   advsecurenet defense adversarial-training -c ./adv_training.yml

Running an evaluation:

.. code:: bash

   advsecurenet evaluate benign -c ./evaluate_benign.yml

   or

   advsecurenet evaluate adversarial -c ./evaluate_adversarial.yml

Python Package
~~~~~~~~~~~~~~

You can import the library as a Python package. You can use the
``advsecurenet`` module to access the library. You can find the
available modules and classes in the
`documentation <http://melihcatal.github.io/advsecurenet/>`__.

|image| *Usage example of AdvSecureNet demonstrating the equivalence
between a YAML configuration file with a command-line interface (CLI)
command and a corresponding Python API implementation.*

Examples
--------

Examples of different use cases can be found in the
`examples <https://github.com/melihcatal/advsecurenet/tree/main/examples/>`__ directory.

Architecture
------------

The high-level architecture of the toolkit is shown in the figure below.

.. figure:: https://drive.switch.ch/index.php/s/SdKAyOZs1d9bcin/download
   :alt: AdvSecureNet API Architecture

   AdvSecureNet API Architecture

.. figure:: https://drive.switch.ch/index.php/s/ZbjIHBHql0dV6n0/download
   :alt: AdvSecureNet CLI Architecture

    AdvSecureNet CLI Architecture

The toolkit is designed to be modular and extensible. CLI and Python API
are implemented separately, however, they share the same core components
and they have the same package structure for the sake of consistency.
Tests are implemented for both CLI and Python API to ensure the
correctness of the implementation and again they follow the same
structure. The toolkit is designed to be easily extensible, new attacks,
defenses, and evaluation metrics can be added by implementing the
corresponding classes and registering them in the corresponding
registries.

Comparison with Other Libraries
-------------------------------

AdvSecureNet stands out among adversarial machine learning toolkits like IBM ART, AdverTorch, SecML, FoolBox, ARES, and CleverHans. Key advantages include:

- **Active Maintenance:** Ensures ongoing support and updates.
- **Comprehensive Training Support:** One of the few toolkits supporting both adversarial and ensemble adversarial training.
- **Multi-GPU Support:** The first toolkit with native multi-GPU support for attacks, defenses, and evaluations, ideal for large-scale experiments.
- **Flexible Interfaces:** The first toolkit that fully supports CLI, API usage, and external YAML configuration files for reproducibility for all features.
- **Performance:** AdvSecureNet excels in performance, significantly reducing execution times on multi-GPU setups. For example, the multi-GPU PGD attack time (107 seconds) is faster than ARES’s best single GPU time (183 seconds). Adversarial training time is reduced from 304 seconds on a single GPU to 166 seconds with 7 GPUs, a speedup of 1.83x.

|performance| *Performance Benchmark for AdvSecureNet and Other Toolkits*
|comparision| *[1] SecML supports attacks from CleverHans and FoolBox [2] This feature is only available for adversarial training.*

License
-------

This project is licensed under the terms of the MIT license. See
`LICENSE <https://github.com/melihcatal/advsecurenet/blob/main/LICENSE>`__ for more details.


.. |Quality Gate Status| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=alert_status
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Bugs| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=bugs
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Code Smells| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=code_smells
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=coverage
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Duplicated Lines (%)| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=duplicated_lines_density
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Lines of Code| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=ncloc
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Reliability Rating| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=reliability_rating
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Security Rating| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=security_rating
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Technical Debt| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=sqale_index
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Maintainability Rating| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=sqale_rating
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Vulnerabilities| image:: https://sonarcloud.io/api/project_badges/measure?project=melihcatal_advsecurenet&metric=vulnerabilities
   :target: https://sonarcloud.io/summary/new_code?id=melihcatal_advsecurenet
.. |Unit Tests and Style Checks| image:: https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml/badge.svg?branch=develop
   :target: https://github.com/melihcatal/advsecurenet/actions/workflows/python-ci.yml
.. |Build and Deploy Sphinx Documentation| image:: https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml/badge.svg
   :target: https://github.com/melihcatal/advsecurenet/actions/workflows/documentation.yml
.. |Upload Python Package| image:: https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/melihcatal/advsecurenet/actions/workflows/python-publish.yml
.. |image| image:: https://github.com/melihcatal/advsecurenet/assets/46859098/f3f86817-8ac3-4523-8f5e-cc9d4b4cbcf3
.. |advsecurenet_logo| image:: https://github.com/melihcatal/advsecurenet/assets/46859098/cdad6b95-5a40-491f-a3d1-c85a3976d681
.. |performance| image:: https://github.com/melihcatal/advsecurenet/assets/46859098/33a4678c-4e22-4dc8-9929-d7c5c2e3c03b
.. |comparision| image:: https://github.com/melihcatal/advsecurenet/assets/46859098/48744f4c-afae-48ff-8c39-2dea55ba8a3a