## AdvSecureNet Architecture

### Overview

AdvSecureNet is a adversarial robustness library for neural networks. It is built on top of [PyTorch](https://pytorch.org/). It is designed to be modular and extensible, allowing for easy experimentation with different adversarial robustness techniques. It is also designed to be easy to use, with a simple API that allows for quick prototyping.

AdvSecureNet adapts a pipeline-based approach to adversarial robustness. This design allows users to construct a sequence of adversarial tasks in a coherent manner, each feeding into the next.

Each step or module in this sequence is independent, meaning it can be customized, replaced, or even bypassed without disrupting the overall workflow. This is particularly beneficial for researchers and practitioners who wish to experiment with various combinations of attacks, defenses, and evaluations without the hassle of reconfiguring the entire setup.

AdvSecureNet's design also promotes reproducibility. By clearly delineating each stage of the adversarial robustness experimentation process, users can effortlessly communicate their methodology and replicate experiments with precision.

### Structure

AdvSecureNet is structured as a sequence of modules, each of which is responsible for a specific task in the adversarial robustness pipeline. The following diagram illustrates the overall structure of AdvSecureNet:
