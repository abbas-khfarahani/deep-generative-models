# Homework 3 — Energy-based models and score networks

Notebooks:

- [ebm_mnist.ipynb](ebm_mnist.ipynb) — energy-based model on MNIST, Langevin sampling, and denoising
- [ncsn.ipynb](ncsn.ipynb) — noise-conditional score network, unconditional and class-conditional

## 1. Energy-based models

An energy-based model writes an unnormalized density as p(x) = exp(f_θ(x)) / Z(θ). The practical difficulty is Z, and the notebook trains the energy with contrastive divergence and samples with Langevin dynamics.

### Training methods for face-scale images

Four standard ways to avoid an exact partition function:

1. **Contrastive divergence.** Replace the model expectation in the likelihood gradient with a short MCMC chain, instead of sampling the model to convergence.
2. **Score matching.** Learn ∇_x log p(x) directly. The score does not depend on Z.
3. **Noise-contrastive estimation.** Train the model to tell real data from noise samples, which also removes Z from the objective.
4. **Persistent contrastive divergence.** Keep the MCMC chains alive across updates, which works better for a more complicated energy.

For images, contrastive divergence and score matching are the usual choices, because they scale to high-dimensional inputs.

### Rejection sampling from a slightly wider Gaussian

Let the target be an unnormalized Gaussian P̃ = N(μ, σ_p² I) and the proposal Q = N(μ, σ_q² I), with σ_q = 1.01 σ_p. Rejection sampling draws x from Q and u from Uniform(0, 1), and accepts x when

```text
u ≤ P̃(x) / (M Q(x))
```

M has to dominate the density ratio. The ratio of the two Gaussians is largest at x = μ, where it equals (1.01)^d, so the smallest valid M is (1.01)^d. The optimal acceptance rate is

```text
1 / M = 1 / (1.01)^d
```

On MNIST, d = 28 × 28 = 784, and that rate is about 0.0004. The proposal is only one percent wider than the target, so almost every draw is rejected. Rejection sampling is not a practical sampler in this regime.

### What the likelihood gradient does to the energy

Gradient ascent on the log-likelihood raises f_θ on training images and lowers f_θ on samples drawn from the model. Because p_θ(x) ∝ exp(f_θ(x)), that increases the probability of the data and decreases the probability of model samples that do not look like data. The update is contrastive: real points are pushed up, model points are pushed down.

The second term is an expectation under p_θ. Computing it exactly means running MCMC to convergence on every update, with many chains, in a high-dimensional space. That cost is paid again at every iteration.

Contrastive divergence with k = 3 cuts the chain short. Start at a training image, take three MCMC steps, and use that third state in place of a true model sample. The estimate is biased, but it is fast enough to train an energy model, and because the chain starts on the data it only has to move a short distance.

### MNIST generation

The notebook shows random training and test digits, then samples drawn by Langevin dynamics from uniform noise in [0, 1], using the model at epoch 10.

Those samples are soft blobs, not digits. Edges are missing, some shapes have several lobes, and a few have holes. Real MNIST digits at the same grid size are sharp and readable. The generated set is not.

Reasons the samples stay poor at this checkpoint:

- Ten epochs is early. The energy has not fit the data distribution, so Langevin dynamics has no sharp modes to fall into. Energy models on images usually need much longer training, on the order of 50–100 epochs.
- The Langevin step size, the number of steps, and the noise schedule may be off. Starting from uniform noise in [0, 1] may also be a poor initialization.
- λ = 10⁻³ and the learning rate may not be the right operating point.
- The network may not have enough capacity, and it may be covering only part of the data (a mild mode collapse), which shows up as low diversity.
- The denoising experiment is consistent with this. At noise 0.2 the model recovers structure, so it has learned coarse shape, but it has not learned the fine detail that unconditional generation needs.

### Denoising

The same energy is used to denoise test digits at three Gaussian noise levels.

| Noise | Result |
| --- | --- |
| 0.2 | Strong. Noise is removed, digits stay sharp, and the result is close to the original. |
| 0.4 | Partial. Most of the noise goes, the digit is still recognizable, but some blur and leftover noise remain. |
| 0.6 | Weak. The image stays degraded, blur and distortion are large, and some digits are hard to read. |

At low noise the original signal is still visible, so the energy can separate it from the noise and the chain has a short path back to the data. At high noise the signal is largely gone. The energy can mistake noise for structure, the chain falls into poor local modes, and no model of this size can invent detail that the noise destroyed.

## 2. Score-based models

### The score does not depend on Z

The score of a density is

```text
s(x) = ∇_x log p(x)
```

It points toward higher probability. For p_θ(x) = exp(−E_θ(x)) / Z(θ),

```text
log p_θ(x) = −E_θ(x) − log Z(θ)
```

Z depends on θ, not on x, so its gradient in x is zero and

```text
s_θ(x) = −∇_x E_θ(x)
```

That is the useful fact. Z is an integral over the whole space and is intractable for images. The score never needs it, so score matching can train the model, and ∇_x E_θ(x) is a normal backpropagation. This is what makes energy models usable in image dimension.

### Why the trace term is replaced by denoising score matching

The original score-matching objective contains tr(∇_x s_θ(x)), the trace of the Hessian of log p_θ. For an image with d pixels that Hessian is d × d. MNIST is 784 × 784, and a 256 × 256 image is far beyond that. Each Hessian entry needs its own backward pass.

Denoising score matching avoids the trace by corrupting the data. Draw x̃ from N(x, σ² I) and train

```text
J_DSM(θ) = E[ || s_θ(x̃) − ∇_{x̃} log q_σ(x̃ | x) ||² ]
```

For Gaussian noise the conditional score is analytic:

```text
∇_{x̃} log q_σ(x̃ | x) = −(x̃ − x) / σ²
```

There is no Hessian. Minimizing this objective is the same as score matching on the noisy marginal q_σ. The trace term in the original objective does not depend on how well s_θ matches the score, so the minimizer is the minimizer of the squared error alone. Matching the Gaussian conditional score, in expectation over the data, matches the score of that noisy marginal.

### A mixture with disjoint support

Let p_data(x) = π p_1(x) + (1 − π) p_2(x), with p_1 supported only on a region A and p_2 only on a disjoint region B.

On A the mixture is π p_1, so

```text
∇_x log p_data(x) = ∇_x log p_1(x)
```

because log π does not depend on x. On B the score is ∇_x log p_2(x). The mixing weight π is invisible to the score inside either region.

Langevin dynamics follows that score plus isotropic noise. Inside A it only sees p_1, so it has no signal that a fraction 1 − π of the mass lives in B, and with a gap between the regions the chance of jumping the gap is essentially zero. Chains get stuck in whichever mode they started in, and the sample histogram does not reproduce π. Annealed or multi-scale noise, as in NCSN, fills the gap so that chains can move between modes before the noise is removed.

### Why a single noise level is not enough, and what NCSN changes

Score matching on the raw data fails for four related reasons.

- Real images lie near a low-dimensional manifold. Off that manifold the density is zero and the score is undefined or unstable.
- Modes are separated by empty space, so a Langevin chain cannot cross from one mode to another.
- Low-density regions have almost no training points, so the learned score there is poor.
- A chain that starts in one of those regions may never reach a mode, and the samples look bad even when the score is right on the data.

NCSN trains one network on a geometric noise ladder σ_1 > σ_2 > … > σ_L. The noisy densities have full support, so the score is defined everywhere, and large noise connects the modes. Sampling is annealed Langevin dynamics: start from N(0, σ_L² I), take Langevin steps at the current noise level, then move to the next smaller noise.

σ_L, the largest noise, has to be wide enough to connect every mode and to cover the data range. If it is too small, the modes stay separated. σ_1, the smallest noise, has to be close to zero so the last distribution is almost the data distribution and fine detail survives. A typical small value is about 0.01. The number of levels has to be large enough that each drop in noise is gentle.

### Unconditional and conditional NCSN

Both models use the same U-Net of adaptive residual blocks, the same Gaussian Fourier embedding of the noise level, and the same FiLM layers to inject that embedding. The noise ladder is shared: σ from 30 down to 0.01, with 20 levels. Both models were trained for 20 epochs.

| | Base | Conditional |
| --- | --- | --- |
| Final loss | 0.058606 | 0.058064 |
| Minimum loss | 0.058568 | 0.058064 |
| Parameters | 2,419,713 | 2,584,833 |

Both start above 0.14 and drop quickly in the first five epochs. After that the curves flatten near 0.058. The conditional model is slightly lower from about epoch 10 onward. The gap is small, but it is in the direction you would expect: the class label is extra information about p(x | y).

The conditional model adds an embedding of the ten digit classes. Inside each residual block that embedding is concatenated with the noise embedding and passed into FiLM. The extra 165,120 parameters are about 6.8% of the base model. They do not change the training cost much, and they are what lets the sampler ask for a specific digit.

The base model draws an unconditional digit and cannot be told which class to produce. The conditional model needs a label at training time and at sampling time, and in return it can draw a requested digit. The notebook shows the training curves, a grid of final samples, and the noise-to-digit trajectory.

Choose the conditional model when the class matters. Choose the base model when any digit is acceptable or when the extra parameters and labels are not worth it. Either model could be pushed further: a longer or better sampler for the base model, and attention or a richer class embedding for the conditional model. The shared U-Net and the FiLM injection are enough for both settings, which is the practical point of the experiment.
