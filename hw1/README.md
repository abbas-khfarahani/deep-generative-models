# Homework 1 — Graphical models and variational autoencoders

Practical code: [vae.ipynb](vae.ipynb). It trains a VAE and two β-VAEs on dSprites, plots the losses, shows reconstructions, computes the mutual information gap, and projects the latent space with PCA.

## 1. Probabilistic graphical models

### 1.1 A small medical Bayesian network

The variables are:

| Symbol | Meaning |
| --- | --- |
| M | Immune-system strength |
| S | Season |
| F | Ability to pay for treatment |
| I | Disease severity |
| T | Drug that is actually used |
| D | Chance of death |

Severity depends on the immune system and on the season. The chosen drug depends on severity and on whether the expensive drug is affordable. Death depends on severity and on the drug.

**Joint distribution**

```text
P(M, I, S, D, T, F) = P(M) P(S) P(F) P(I | M, S) P(T | I, F) P(D | T, I)
```

**Conditional independence**

| Claim | Verdict | Reason |
| --- | --- | --- |
| F ⊥ D | False | The path F → T → D is open. The claim is not conditioned on T. |
| S ⊥ D \| I | True | Conditioning on I blocks the path from season to death (d-separation). |
| M ⊥ F | True | M and F meet at T in a v-structure, and T is not observed, so the path is blocked. |
| M ⊥ F \| T | False | Observing T opens that v-structure, so M and F become dependent. |
| M ⊥ T \| {D, I} | False | Conditioning on I blocks the direct path M → I → T, but conditioning on D opens the v-structure between I and T. M is a parent of I, so the path stays active. |

### 1.2 Bayesian graph, moral graph, and variable elimination

The Bayesian graph has an isolated node C, and the remaining edges are O → S, O → T, A → T, S → B, T → M, A → M, and B → M.

**Joint**

```text
P(A, B, C, O, S, T, M)
  = P(C) P(A) P(O) P(T | O, A) P(S | O) P(B | S) P(M | A, B, T)
```

**Markov blanket of T**

In a Bayesian network the blanket of a node is its parents, its children, and the other parents of its children.

```text
MB(T) = {A, O, M, B}
```

A and O are the parents of T. M is the child of T. B is another parent of M.

**Is the drawn Markov graph a perfect I-map of this Bayesian network?**

No. The Bayesian network has a v-structure whose independence is not kept by the undirected graph: the moralized graph connects nodes that were independent in the directed graph.

**Is that Markov graph chordal?**

No. It contains a cycle of length greater than 3 with no chord.

**Joint of the Markov graph, as a product of clique potentials**

```text
P(A, B, C, O, S, T, M)
  ∝ φ(C) · φ(S, B) · φ(T, M, B) · φ(O, A, T) · φ(A, T, M)
```

**Removing C**

Both claims hold.

- On the Bayesian graph, C is independent of every other variable, so dropping the factor p(C) leaves the joint over the remaining variables.
- On the Markov graph, C appears only in φ(C). If that potential is already normalized, ∫ φ(C) dC = 1, dropping it does not change the normalizer of the remaining distribution.

### 1.3 Markov network on A–G

The undirected graph has edges A–B, A–C, B–C, B–D, B–E, C–F, D–F, E–F, and E–G.

The maximal cliques are {A, B, C}, {B, D}, {B, E}, {C, F}, {D, F}, {E, F}, and {E, G}, so

```text
P(A, B, C, D, E, F, G)
  ∝ φ(A, B, C) φ(B, D) φ(B, E) φ(C, F) φ(D, F) φ(E, F) φ(E, G)
```

**Conditional independence**

| Claim | Verdict | Reason |
| --- | --- | --- |
| A ⊥ G | False | A–B–E–G is an open path. |
| F ⊥ A \| {D, C} | False | A–B–E–F does not go through D or C, so it stays open. |
| G ⊥ C \| E | True | Every path from G to the rest of the graph passes through E. |
| P(A \| B, C) = P(A \| B, C, E) | True | Given B and C, A is independent of E, so adding E to the condition does not change the distribution. |

**Scaling a potential**

Replacing φ(E, G) by 5 φ(E, G) does not change the distribution. The partition function Z grows by the same factor, and a probability distribution stays normalized.

### 1.4 Variational inference on an exponential model

The submitted report does not include a written answer for the exponential-family variational problem (the posterior of z given x, with the mean-field family q(z) = θ² z e^(−θz)).

## 2. Variational autoencoder

### Loss

The training objective has two parts.

- The reconstruction term is a log-likelihood. It pushes the encoder to produce a latent code from which the decoder can rebuild the input.
- The KL term measures the gap between the encoder's approximate posterior and the prior. It keeps the latent distribution close to that prior.

### Data

The practical part uses dSprites: simple squares, ellipses, and hearts at different positions, scales, and orientations. The set has about 700,000 images. Samples and reconstructions are in the notebook.

### Reparameterization

The ELBO is not differentiable through a direct sample from the encoder's Gaussian. The reparameterization trick fixes that. After the encoder predicts a mean and a standard deviation, a sample is drawn from a standard normal and then shifted and scaled:

```text
z = μ + σ ⊙ ε,    ε ~ N(0, I)
```

The randomness stays in ε, and the gradient can pass through μ and σ.

### What β changes

A β-VAE multiplies the KL term by β. A larger β pulls the latent distribution harder toward the prior. That pressure makes the latent factors more independent, which is the disentangling effect, and it can shrink the useful capacity of the latent code. Reconstruction quality often drops because the model is less free to use the latent dimensions however it wants.

The notebook trains β = 2 and β = 10, plots both objectives, and shows eight reconstructions from each model.

### Mutual information gap

A higher MIG means the latent dimensions are more cleanly tied to single factors of variation. A β-VAE with a larger β is expected to score a higher MIG. The per-factor MIG also shows which factors each model captured. The notebook computes the score and shows a PCA view of one latent space.

### VQ-VAE

A VQ-VAE replaces the continuous latent code with a quantized one. Each encoder vector is snapped to the nearest entry in a learned codebook. The latent space is no longer a continuous Gaussian, as in a standard VAE, and the discrete codes can separate features more sharply.

The submitted report does not include the later VQ-VAE implementation subsections beyond this comparison.
