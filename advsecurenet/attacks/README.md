# Adversarial Attacks

AdvSecureNet supports various adversarial attacks, including:

- Fast Gradient Sign Method (FGSM)
- Carlini and Wagner (C&W)
- Projected Gradient Descent (PGD)
- DeepFool
- Decision Boundary
- Layerwise Origin-Target Synthesis (LOTS)

Some of these attacks are targeted, while others are untargeted. AdvSecureNet provides a simple way to use targeted adversarial attacks by having an automatic target generation mechanism. This mechanism generates a target label that is different from the original label of the input image. The target label is chosen randomly from the set of possible labels, excluding the original label. This ensures that the attack is targeted, as the goal is to mislead the model into predicting the target label instead of the correct one without being explicitly specified by the user. This feature is particularly useful for the large datasets where manually specifying target labels for each input image is impractical. However, users can also specify the target label manually if they would like to do so.

Below, we provide a brief overview of each adversarial attack supported by AdvSecureNet, including its characteristics, purpose, and potential applications.

## Adversarial Attacks Overview

Adversarial attacks can be categorized in different ways. One way to categorize them is based on the **information** they use. Broadly, these attacks fall into two categories: **white-box** and **black-box** attacks[^1][^2]. White-box attacks necessitate access to the model's parameters, making them intrinsically reliant on detailed knowledge of the model's internals[^1][^2]. In contrast, black-box attacks operate without requiring access to the model's parameters[^1][^2]. Among black-box attack methods, one prevalent approach involves training a substitute model to exploit the transferability of adversarial attacks[^3], targeting the victim model indirectly[^4]. Additionally, there are other black-box attack methods such as decision boundary attacks[^5] and zeroth order optimization based attacks such as ZOO[^6]. These methods, distinct from the substitute model approach, rely solely on the output of the model, further reinforcing their classification as black-box attacks.

Adversarial attacks can also be differentiated based on the **number of steps** involved in generating adversarial perturbations. This categorization divides them into **single-step** and **iterative** attacks[^1]. Single-step attacks are characterized by their speed, as they require only one step to calculate the adversarial perturbation. On the other hand, iterative attacks are more time-consuming, involving multiple steps to incrementally compute the adversarial perturbation[^1].

Additionally, another classification of adversarial attacks hinges on the **objective of the attack**. In this context, attacks are grouped into **targeted** and **untargeted** categories[^1]. Targeted attacks are designed with the specific goal of manipulating the model's output to a predetermined class. In contrast, untargeted attacks are aimed at causing the model to incorrectly classify the input into any class, provided it is not the correct one[^1].

---

### FGSM

FGSM, short for Fast Gradient Sign Method, is a type of adversarial attack that was introduced by Goodfellow et al.[^7] in 2015. It is a single-step, white box attack. Initially, the attack is designed as an untargeted attack. However, it can be modified to be a targeted attack. The idea of the FGSM attack is to compute the adversarial perturbation by taking the sign of the gradient of the loss function with respect to the input. The FGSM attack is a fast attack since it only requires one step to compute the adversarial perturbation. This makes it a popular attack in the adversarial robustness literature. However, it has been shown that the FGSM attack is not effective against the adversarial defenses. This is because the FGSM attack is a weak attack and it can be easily defended by the adversarial defenses such as adversarial training.

If the attack is untargeted, the formula tries to maximize the loss function with respect to the input and correct label. If the attack is targeted, the formula tries to minimize the loss function with respect to the input and target label since the purpose of the targeted attack is to get closer to the target label. The untargeted FGSM attack is given in the equation below, and the targeted FGSM attack is given in the subsequent equation.

$$
\text{adv}_x = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

$$
\text{adv}_x = x - \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

where:

- $\text{adv}_x$ is the adversarial image.
- $x$ is the original input image.
- $y$ is the original label for **untargeted** attacks, or the target label for **targeted** attacks.
- $\epsilon$ is a multiplier to ensure the perturbations are small.
- $\theta$ represents the model parameters.
- $J(\theta, x, y)$ is the loss function used by the model.

---

### C&W

The Carlini and Wagner (C&W) attack[^8], introduced by Nicholas Carlini and David Wagner in 2017, is a sophisticated method of adversarial attack aimed at machine learning models, particularly those used in computer vision. As an iterative, white-box attack, it requires access to the model's architecture and parameters. The core of the C&W attack involves formulating and solving an optimization problem that minimally perturbs the input image in a way that leads to incorrect model predictions. This is done while maintaining the perturbations imperceptible to the human eye, thus preserving the image's visual integrity. The optimization process often employs techniques like binary search to find the smallest possible perturbation that can deceive the model. The C&W attack is versatile, capable of being deployed as both untargeted and targeted attacks. The attack can also use different distance metrics when computing the perturbation, such as the L0, L2, and L-infinity norms. The choice of distance metric can affect the attack's effectiveness and the perturbation's perceptibility.

The attack's effectiveness lies in its ability to subtly manipulate the input data, challenging the robustness and security of machine learning models, and it has become a benchmark for testing the vulnerability of these models to adversarial examples. However, the biggest drawback of the C&W attack is its computational complexity, which stems from the iterative nature of the attack and the optimization problem that needs to be solved. This makes the C&W attack less practical for real-world applications.

$$
\begin{aligned}
& \text{minimize} \quad \|\delta\|_p + c \cdot f(x + \delta) \\
& \text{such that} \quad x + \delta \in [0,1]^n
\end{aligned}
$$

where:

- $\delta$ is the perturbation added to the input image $x$.
- $\|\delta\|_p$ is the p-norm of the perturbation, which measures the size of the perturbation.
- $c$ is a constant that balances the perturbation magnitude and the success of the attack.
- $f(x + \delta)$ is the objective function, designed to mislead the model into making incorrect predictions.
- $x + \delta \in [0,1]^n$ ensures that the perturbed input remains within the valid input range for the model.

---

### PGD

The Projected Gradient Descent (PGD) attack[^9] is a prominent adversarial attack method in the field of machine learning, particularly for evaluating the robustness of models against adversarial examples. Introduced by Madry et al.[^9], the PGD attack is an iterative method that generates adversarial examples by repeatedly applying a small perturbation and projecting this perturbation onto an $\varepsilon$-ball around the original input within a specified norm. This process is repeated for a fixed number of steps or until a successful adversarial example is found. The PGD attack operates under a white-box setting, where the attacker has full knowledge of the model, including its architecture and parameters. The strength of the PGD attack lies in its simplicity and effectiveness in finding adversarial examples within a constrained space, making it a standard benchmark in adversarial robustness research. According to cite here, the PGD attack is one of the most used attacks in adversarial training. However, similar to other adversarial attacks like the C&W attack, the PGD attack can be computationally intensive, particularly when dealing with complex models and high-dimensional input spaces, which may limit its practicality in real-world scenarios.

---

### DeepFool

The DeepFool attack, introduced by Moosavi-Dezfooli et al.[^10] in 2016, is a type of adversarial attack that aims to generate adversarial examples that are close to the original input but mislead the model. It is an iterative, white-box attack. The algorithm works by linearizing the decision boundaries of the model and then applying a small perturbation that pushes the input just across this boundary. This process is repeated iteratively until the input is misclassified, ensuring that the resulting adversarial example is as close to the original input as possible. One of the key strengths of DeepFool is its ability to compute these minimal perturbations with relatively low computational overhead compared to other methods because of its linearization approach. Despite its efficiency, the attack assumes a somewhat idealized linear model, which may not always accurately reflect the complex decision boundaries in more advanced, non-linear models. Nonetheless, DeepFool has become a valuable tool in the adversarial machine learning toolkit for its ability to provide insights into model vulnerabilities with minimal perturbations.

---

### Decision Boundary

The Decision Boundary attack is a black-box attack that was introduced by Brendel et al.[^11] in 2017. The idea of the Decision Boundary attack is to find the decision boundary of the model and then apply a small perturbation that pushes the input just across this boundary. The Decision Boundary attack is an iterative attack and can be both targeted and untargeted. The attack starts with a random input that is initially adversarial and then iteratively updates the input to get closer to the decision boundary and minimize the perturbation. The advantage of the attack is that it does not require any information about the model. This makes it more suitable for real-world applications where the model's information is not available. However, the drawback of the attack is that it is computationally expensive since it requires iteratively updating the input to get closer to the decision boundary.

---

### LOTS

LOTS, Layerwise Origin-Target Synthesis[^12], is a type of adversarial attack that was introduced by Rozsa et al. in 2017. It is a versatile, white-box attack that can be used as both targeted and untargeted attacks, single-step and iterative. The idea of the LOTS attack is to compute the adversarial perturbation by using the deep feature layers of the model. The purpose of the attack algorithm is to adjust the deep feature representation of the input to match the deep feature representation of the target class. Utilizing deep feature representations makes the LOTS attack suitable for systems that use deep feature representations, such as face recognition systems. The results show that the Iterative LOTS attack is highly successful against the VGG Face network with success rates between 98.28% to 100%[^12]. However, the drawback of the LOTS attack is that it needs to know the deep feature representation of the target class.

---

### References

[^1]: Khalid, F., Hanif, M. A., & Shafique, M. (2021). Exploiting Vulnerabilities in Deep Neural Networks: Adversarial and Fault-Injection Attacks. arXiv preprint arXiv:2105.03251.
[^2]: Chakraborty, A., Alam, M., Dey, V., Chattopadhyay, A., & Mukhopadhyay, D. (2018). Adversarial Attacks and Defences: A Survey. arXiv preprint arXiv:1810.00069.
[^3]: Papernot, N., Mcdaniel, P., & Goodfellow, I. J. (2016). Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples. arXiv preprint arXiv:1605.07277. Retrieved from https://api.semanticscholar.org/CorpusID:17362994
[^4]: Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). Practical Black-Box Attacks against Machine Learning. arXiv preprint arXiv:1602.02697.
[^5]: Brendel, W., Rauber, J., & Bethge, M. (2017). Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models. arXiv preprint arXiv:1712.04248. Retrieved from https://api.semanticscholar.org/CorpusID:2410333
[^6]: Chen, P.-Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C.-J. (2017). ZOO: Zeroth Order Optimization Based Black-Box Attacks to Deep Neural Networks without Training Substitute Models. In _Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security_ (pp. 15-26). New York, NY, USA: Association for Computing Machinery. doi: [10.1145/3128572.3140448](https://doi.org/10.1145/3128572.3140448)
[^7]: Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. arXiv preprint arXiv:1412.6572.
[^8]: Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." Available at: [arXiv:1702.04267](https://arxiv.org/abs/1702.04267)
[^9]: Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). "Towards Deep Learning Models Resistant to Adversarial Attacks." Available at: [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
[^10]: Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016). "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks." Available at: [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)
[^11]: Brendel, W., Rauber, J., & Bethge, M. (2017). "Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models." Available at: [arXiv:1712.04248](https://arxiv.org/abs/1712.04248)
[^12]: Rozsa, A., Rudd, E. M., & Boult, T. E. (2017). "LOTS: Layerwise Origin-Target Synthesis for Adversarial Attack." Available at: [arXiv:1611.06503](https://arxiv.org/abs/1611.06503)
