Advsecurenet supports multiple evaluation methods.

## Attack Success Rate

Attack Success Rate (ASR) is the fundamental metric for evaluating the adversarial robustness of the model. It is the percentage of the adversarial examples that are successfully misclassified by the model. In the context of adversarial attacks, success can be defined differently for targeted and untargeted attacks. For targeted attacks, success means that the adversarial example is classified as a specific target class. For untargeted attacks, success means that the adversarial example is classified as any class other than the true class. The higher the ASR, the more vulnerable the model is to adversarial attacks. The ASR is given in the equation below.

## Attack Success Rate

Attack Success Rate (ASR) is the fundamental metric for evaluating the adversarial robustness of the model. It is the percentage of the adversarial examples that are successfully misclassified by the model. In the context of adversarial attacks, success can be defined differently for targeted and untargeted attacks. For targeted attacks, success means that the adversarial example is classified as a specific target class. For untargeted attacks, success means that the adversarial example is classified as any class other than the true class. The higher the ASR, the more vulnerable the model is to adversarial attacks. The ASR is given by the equation below.

$$
ASR = \frac{\text{Number of Successful Adversarial Examples}}{\text{Total Number of Adversarial Examples}}
$$

where:

- **Number of Successful Adversarial Examples** refers to the count of adversarial examples that lead to a successful attack, where success is defined as:
  - In targeted attacks, the adversarial example is classified as the target class, _y~target~_.
  - In untargeted attacks, the adversarial example is classified as any class other than the true class, _y~true~_.
- **Total Number of Adversarial Examples** is the total number of adversarial examples generated from inputs where the model initially makes a correct prediction.

## Adversarial Transferability Evaluation

The evaluation of adversarial transferability is facilitated by the `transferability_evaluator` class within the `advsecurenet.evaluation.evaluators` module. This evaluator is designed to assess the effectiveness of adversarial examples, originally generated for a source model, in deceiving various target models.

The evaluator is initialized with a list of target models. During the evaluation phase, the `update` method calculates whether the adversarial examples successfully mislead both the source and the target models. Success in targeted attacks is determined by the adversarial example being classified as the target class, while in untargeted attacks, success is achieved if the example is classified as any class other than its true class.

The evaluator maintains a tally of successful deceptions for each target model, as well as the total count of adversarial examples that successfully deceive the source model. The transferability rate for each target model is calculated as follows:

$$
Transferability\ Rate = \frac{\text{Number of Successful Transfers to Target Model}}{\text{Total Number of Successful Adversarial Examples on Source Model}}
$$

where:

- **Number of Successful Transfers to Target Model** is the count of adversarial examples that successfully deceive the target model.
- **Total Number of Successful Adversarial Examples on Source Model** refers to the count of adversarial examples that initially misled the source model.

This evaluation method provides a thorough analysis of the transferability of adversarial examples across different models, shedding light on the robustness of each model against such attacks.

> **Example:**
>
> Consider a scenario with a source model, Model_A, and two target models, Model_B and Model_C. Suppose Model_A generates 100 adversarial examples, out of which 80 successfully deceive Model_A. When these 80 adversarial examples are tested against Model_B, 50 are successful, and against Model_C, 30 are successful.
>
> Using the transferability rate formula:
>
> For Model_B:
>
> $$
> Transferability\ Rate_{Model_B} = \frac{50}{80} = 0.625
> $$
>
> For Model_C:
>
> $$
> Transferability\ Rate_{Model_C} = \frac{30}{80} = 0.375
> $$
>
> This indicates that adversarial examples from Model_A are more transferable to Model_B than to Model_C.

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/17aae7a4-3af7-4585-aaaf-a2c520cbe82c)
_Example of Adversarial Transferability. Taken from [Closer Look at the Transferability of Adversarial Examples: How They Fool Different Models Differently](https://arxiv.org/abs/2112.14337)_

## Robustness Gap

Robustness Gap is a metric that measures the difference between the accuracy on clean examples and the accuracy on adversarial examples. The higher the robustness gap is, the more vulnerable the model is to adversarial attacks. Possible values for the robustness gap are between 0 and 1. 0 means that the model performs the same on clean and adversarial examples. 1 means that the model performs perfectly on clean examples but completely fails on adversarial examples.

**Clean Accuracy** ($A_{\text{clean}}$) is calculated as the ratio of the total number of correctly classified clean images ($N_{\text{correct\_clean}}$) to the total number of samples ($N_{\text{total}}$):

$$
A_{\text{clean}} = \frac{N_{\text{correct\_clean}}}{N_{\text{total}}}
$$

**Adversarial Accuracy** ($A_{\text{adv}}$) is the ratio of the total number of correctly classified adversarial images ($N_{\text{correct\_adv}}$) to the total number of samples ($N_{\text{total}}$):

$$
A_{\text{adv}} = \frac{N_{\text{correct\_adv}}}{N_{\text{total}}}
$$

**The Robustness Gap** ($G_{\text{robust}}$) is the difference between Clean Accuracy and Adversarial Accuracy:

$$
G_{\text{robust}} = A_{\text{clean}} - A_{\text{adv}}
$$

Evaluator for the perturbation effectiveness. The effectiveness score is the attack success rate divided by the perturbation distance. The higher the score, the more effective the attack.

> **Example:**
>
> Suppose we have a model tested on a dataset of 1000 images. Out of these, the model correctly classifies 950 clean images, giving us $N_{\text{correct\_clean}} = 950$ and $N_{\text{total}} = 1000$. When exposed to adversarial examples, the model correctly classifies only 700 of these images, thus $N_{\text{correct\_adv}} = 700$.
>
> Using the provided formulas, we can calculate the Clean Accuracy ($A_{\text{clean}}$) and the Adversarial Accuracy ($A_{\text{adv}}$):
>
> $$
> A_{\text{clean}} = \frac{950}{1000} = 0.95
> $$
>
> $$
> A_{\text{adv}} = \frac{700}{1000} = 0.70
> $$
>
> Then, the Robustness Gap ($G_{\text{robust}}$) is calculated as:
>
> $$
> G_{\text{robust}} = A_{\text{clean}} - A_{\text{adv}} = 0.95 - 0.70 = 0.25
> $$
>
> This Robustness Gap of 0.25 indicates that the model's performance significantly degrades when exposed to adversarial examples, revealing a vulnerability to such attacks.

## Perturbation Effectiveness

Perturbation Effectiveness is a metric for evaluating the effectiveness of the adversarial perturbation. It is the percentage of the adversarial perturbation that is effective in changing the model's prediction. The higher the perturbation effectiveness is, the more effective the adversarial perturbation is. The purpose of this metric is to distinguish between attacks that have a high success rate but require a large perturbation magnitude, and attacks that have a lower success rate but require a smaller perturbation magnitude. The perturbation effectiveness is given by the equation below.

$$
PE = \frac{\text{Attack Success Rate}}{\text{Perturbation}}
$$

where:

- **Attack Success Rate** is the percentage of the adversarial examples that are successfully misclassified by the model.
- **Perturbation** is the perturbation magnitude of the adversarial examples. It can be measured using different norms, such as L1, L2, or Linf.

## Similarity Metrics

### PSNR

The Peak Signal-to-Noise Ratio (PSNR) metric is a standard used in the field of image processing for assessing the quality of reconstructed or compressed images in relation to the original ones. The PSNR is derived from the mean squared error (MSE) between the original image and the reconstructed one. It is typically expressed in decibels (dB), indicating the ratio of the maximum possible power of a signal to the power of corrupting noise.

The formula for PSNR is given by:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
$$

where $\text{MAX}_I$ represents the maximum possible pixel value of the image (e.g., 255 for 8-bit images), and MSE is the mean squared error between the original and reconstructed images.

The range of PSNR is typically between 0 dB to infinity, with higher values indicating a smaller difference between the original and reconstructed image, and thus, better quality. In cases where the original and reconstructed images are identical, the MSE becomes zero, leading to an undefined PSNR in the logarithmic scale, which can be theoretically considered as infinite. A higher PSNR value generally suggests that the reconstructed image closely resembles the original image in quality.

### SSIM

The Structural Similarity Index Measure (SSIM) is a metric used for measuring the similarity between two images. Unlike traditional methods like PSNR that focus on pixel-level differences, SSIM considers changes in structural information, luminance, and contrast, providing a more perceptually relevant assessment of image quality.

The formula for SSIM is given by:

$$
\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

where $x$ and $y$ are the two images being compared, $\mu_x$, $\mu_y$ are the average pixel values, $\sigma_x^2$, $\sigma_y^2$ are the variances, $\sigma_{xy}$ is the covariance of the images, and $C_1$, $C_2$ are constants used to stabilize the division.

The SSIM index is a decimal value between -1 and 1, where a value of 1 for SSIM implies no difference between the compared images. As the value decreases, the differences between the images increase. SSIM is particularly useful in contexts where a human observer's assessment of quality is important, as it aligns more closely with human visual perception than metrics based solely on pixel differences, which makes it suitable for adversarial robustness evaluation.

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/ac7344d0-cef0-4ec3-b784-9aec32a0c80d)
_Comparison of SSIM and PSNR Metrics vs. Epsilon in FGSM Attack_
![image](https://github.com/melihcatal/advsecurenet/assets/46859098/b8e3d753-5bf5-4963-9f1b-b36df3acae5d)
_SSIM and PSNR Example. Taken from [Medium](https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee)_
