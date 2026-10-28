# Deep Generative Models

Coursework for **Deep Generative Models** at the University of Tehran, School of Electrical and Computer Engineering.

**Abbas Khoshdooni Farahani**

Each homework has an English write-up and the notebooks used for the practical parts. The notes follow the submitted solutions: the reasoning, the numbers, and the conclusions are the ones from those reports.

| Homework | Topics | Folder |
| --- | --- | --- |
| 1 | Bayesian and Markov networks, variational autoencoders on dSprites | [hw1](hw1/) |
| 2 | Masked autoregressive flow, anomaly detection, CycleGAN | [hw2](hw2/) |
| 3 | Energy-based models and noise-conditional score networks on MNIST | [hw3](hw3/) |
| 4 | DDPM and DDIM on FashionMNIST, flow matching on SPY returns, DreamBooth | [hw4](hw4/) |

The notebooks were originally run in Google Colab. Saved cell outputs are kept where the submission included them, so the plots and samples can be read without retraining. Training still needs a GPU for the larger models.

```bash
pip install -r requirements.txt
```

`diffusers`, `transformers`, `accelerate`, and `peft` are only needed for the DreamBooth notebook.
