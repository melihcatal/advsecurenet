AdvSecureNet
============

.. image:: https://drive.switch.ch/index.php/s/DAaKZEh9OeuvTEQ/download
   :width: 400
   :alt: advsecurenet
   :align: center

AdvSecureNet - Adversarial Secure Networks - is a Python library for Machine Learning Security. It has been developed by `Melih Catal <https://github.com/melihcatal>`_ at the `University of Zurich <https://www.uzh.ch/en.html>`_ as a part of his Master's Thesis under the supervision of `Prof. Dr. Manuel Günther <https://www.ifi.uzh.ch/en/aiml/people/guenther.html>`_.

The library provides a set of tools to generate adversarial examples and to evaluate the robustness of machine learning models against adversarial attacks. It also provides a set of tools to train robust machine learning models. The library is built on top of `PyTorch <https://pytorch.org/>`_. It is designed to be modular and extensible. So, anyone can easily run experiments with different configurations.

Installation
============

The library can be installed using `pip <https://pip.pypa.io/en/stable/>`_. You can install the library using the following command:

.. code-block:: bash

    pip install advsecurenet

You can also install the library from the source code. You can clone the repository and install the library using the following commands:

.. code-block:: bash

    git clone
    cd advsecurenet
    pip install -e .

Usage
========= 

The library can be used as a command line tool or as an importable Python package.


Command Line Tool
~~~~~~~~~~~~~~~~~

`advsecurenet` command can be used to interact with the library. You can use `advsecurenet --help` to see the available commands and options. Available commands are:

- `attack` Command to execute attacks.
- `config-default` Generate a default configuration file based on the name...
- `configs` Return the list of available configuration files.
- `defense` Command to execute defenses.
- `model-layers` Command to list the layers of a model.
- `models` Command to list available models.
- `test` Command to evaluate a model.
- `train` Command to train a model.
- `weights` Command to model weights.

You can use `advsecurenet <command> --help` to see the available options for a command. For example, you can use `advsecurenet attack --help` to see the available options for the `attack` command. The CLI supports both config yml files and arguments.


Python Package
~~~~~~~~~~~~~~

You can import the library as a Python package. You can use the `advsecurenet` module to access the library. You can find the available modules and classes in this documentation.

Examples
========= 

You can find various examples in the examples directory. The examples show different use cases of the library and how to use the library as a Python package/CLI tool.

Architecture
============

The high-level architecture of the library is shown in the figure below.

.. image:: https://drive.switch.ch/index.php/s/7dkGCqtlf4uWoai/download
   :width: 400
   :alt: advsecurenet
   :align: center


License
=======
The library is licensed under the  `MIT License <https://opensource.org/licenses/MIT>`_. You can find the license file in the repository.