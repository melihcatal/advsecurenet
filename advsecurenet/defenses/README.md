# Adversarial Training

Adversarial training is a methodology in machine learning, particularly within the field of deep learning, aimed at improving the robustness and generalization of models against adversarial examples. These are inputs deliberately crafted to deceive models into making incorrect predictions or classifications[^1]. The concept of adversarial training emerged as a critical response to the observation that neural networks, despite their high accuracy, are often vulnerable to subtly modified inputs that are imperceptible to humans[^2].

The core idea behind adversarial training involves the intentional generation of adversarial examples during the training process. By exposing the model to these challenging scenarios, the model learns to generalize better and becomes more resistant to such attacks[^3]. So far, adversarial training represents the only known defense that works to some extent and scale against adversarial attacks[^4].

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/da22a465-03b5-4583-97ac-fded10999073)
_Adversarial Training Flow_

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/4ac0a8cb-a97f-4363-b05e-06444f7cd0f5)
_Adversarial Data Generator_

## Ensemble Adversarial Training

Ensemble Adversarial Training, proposed by Florian Tramèr et al.[^5] in 2018, is a type of adversarial training that aims to improve the robustness of the model to unseen attacks and black-box attacks by generalizing the adversarial training process. The idea of the ensemble adversarial training is crafting adversarial examples from a set of pretrained substitute models in addition to the adversarial examples crafted from the original source model that the defender wants to robustify. The intuition is that crafting adversarial samples only from the source model can lead to overfitting to the source model and the model still can be vulnerable to unseen attacks and black-box attacks. However, crafting adversarial samples from a set of pretrained substitute models can lead to generalization of the adversarial training process and improve the robustness of the model to unseen attacks and black-box attacks. The experiments[^5] showed that the ensemble adversarial training can improve the robustness of the model to unseen attacks and black-box attacks but lower the accuracy on clean examples.

The ensemble feature of the ensemble adversarial training refers to ensemble of models. However, it is also possible to ensemble the adversarial attacks[^5]. The intuition is similar to the ensemble of models, which is generalization since the adversarial training does not offer a guarantee to the unseen attacks [^6][^7] . It's also shown that having a robust model to one type of attack can make the model more vulnerable to other types of attacks [^7][^8].

The ensemble of adversarial attacks refers to crafting adversarial examples from a set of adversarial attacks in addition to the adversarial examples crafted from the original adversarial attack. The purpose is having a robust model to different types of perturbations simultaneously [^5]. However, the results show that the models trained with ensemble of adversarial attacks are not robust as the models trained with each attack individually [^5].

![image](https://github.com/melihcatal/advsecurenet/assets/46859098/23601123-2566-468b-b0c5-b1eaa0bf8069)
_Ensemble Adversarial Generator. The generator randomly picks one model from the models pool and one attack from the attacks pool. Only having origin model and one attack is the same as classical adversarial training. Having one attack but multiple pretrained models is the Ensemble Adversarial Training. It's also possible have ensemble models and ensemble attacks at the same time._

## References

[^1]: Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. arXiv preprint arXiv:1412.6572.
[^2]: Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. J., & Fergus, R. (2013). Intriguing properties of neural networks. CoRR, abs/1312.6199. Available at: https://api.semanticscholar.org/CorpusID:604334
[^3]: Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards Deep Learning Models Resistant to Adversarial Attacks. ArXiv, abs/1706.06083. Available at: https://api.semanticscholar.org/CorpusID:3488815
[^4]: Carlini, N., Athalye, A., Papernot, N., Brendel, W., Rauber, J., Tsipras, D., Goodfellow, I., Madry, A., & Kurakin, A. (2019). On Evaluating Adversarial Robustness. arXiv preprint arXiv:1902.06705.
[^5]: Tramer, F., Kurakin, A., Papernot, N., Boneh, D., & Mcdaniel, P. (2017). Ensemble Adversarial Training: Attacks and Defenses. ArXiv, abs/1705.07204. Available at: https://api.semanticscholar.org/CorpusID:21946795
[^6]: Papernot, N., Mcdaniel, P., & Goodfellow, I. J. (2016). Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples. ArXiv, abs/1605.07277. Available at: https://api.semanticscholar.org/CorpusID:17362994
[^7]: Schott, L., Rauber, J., Bethge, M., & Brendel, W. (2018). Towards the first adversarially robust neural network model on MNIST. arXiv preprint arXiv:1805.09190.
[^8]: Engstrom, L., Tsipras, D., Schmidt, L., & Madry, A. (2017). A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations. ArXiv, abs/1712.02779. Available at: https://api.semanticscholar.org/CorpusID:21929206
