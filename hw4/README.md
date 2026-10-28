# Homework 4 — Diffusion, flow matching, and DreamBooth

Notebooks:

- [diffusion_fashionmnist.ipynb](diffusion_fashionmnist.ipynb) — DDPM and DDIM on FashionMNIST, with FID
- [flow_matching_spy.ipynb](flow_matching_spy.ipynb) — conditional flow matching on SPY log returns
- [dreambooth_lora.ipynb](dreambooth_lora.ipynb) — DreamBooth LoRA scaffold

The written solution below covers the diffusion and flow-matching questions. The DreamBooth notebook is the assignment scaffold. Instance loading, the training forward pass, and prior preservation are still marked `TODO` in that file, and the report did not include a separate DreamBooth discussion.

## 1. Denoising diffusion

### Closed form of the forward process

The forward step is

```text
q(x_t | x_{t-1}) = N(√α_t x_{t-1}, (1 − α_t) I),    α_t = 1 − β_t
```

with ᾱ_t = ∏_{s=1}^{t} α_s. The claim is

```text
q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I)
```

At t = 1, ᾱ_1 = α_1, so the closed form is the definition. Assume it holds at t − 1:

```text
x_{t-1} = √ᾱ_{t-1} x_0 + √(1 − ᾱ_{t-1}) ε_{t-1}
```

Substitute into the one-step update x_t = √α_t x_{t-1} + √(1 − α_t) ε_t:

```text
x_t = √(α_t ᾱ_{t-1}) x_0
    + √(α_t (1 − ᾱ_{t-1})) ε_{t-1}
    + √(1 − α_t) ε_t
```

Both noise terms have mean zero and are independent of x_0, and ᾱ_t = α_t ᾱ_{t-1}, so the conditional mean is √ᾱ_t x_0. The conditional variance is the sum of the two noise variances:

```text
α_t (1 − ᾱ_{t-1}) + (1 − α_t) = 1 − ᾱ_t
```

So q(x_t | x_0) is the claimed Gaussian, and the induction is done.

### Why the network predicts the noise

The reverse step can be parameterized as a prediction of x_0, of the reverse-step mean μ_θ, or of the noise ε. Predicting ε is the standard choice for two reasons.

The target ε is standard normal: zero mean, identity covariance, the same scale in every coordinate. The variational bound then collapses, up to terms that do not depend on θ, to

```text
L_simple = E[ || ε − ε_θ(x_t, t) ||² ]
```

Predicting x_0 or μ_θ instead asks the network to match a target that has the structure of a real image, which is a harder, worse-conditioned regression.

The parameterization is also lossless. From the closed form,

```text
x_0 = (x_t − √(1 − ᾱ_t) ε) / √ᾱ_t
```

and the reverse mean μ_θ is a linear function of the same ε_θ. One noise network is enough to recover x_0, to run the DDPM reverse step, and to switch to other targets such as v-prediction.

### The variational bound

The training goal is log p_θ(x_0). It is intractable, so DDPM maximizes a lower bound:

```text
L_VLB = E_q[
  − log p_θ(x_0 | x_1)
  + Σ_{t=2}^{T} KL( q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t) )
  − log p(x_T)
]
```

The first term is the final reconstruction. Each KL compares the true Gaussian forward posterior with the model's reverse Gaussian. When that reverse Gaussian is written with mean μ_θ reparameterized by ε_θ, the θ-dependent part of each KL is a squared error ||ε − ε_θ(x_t, t)||². Minimizing the simple noise loss is therefore minimizing the part of the bound that the network can actually change.

At t near T both distributions are almost pure noise, so those KL terms are small. At small t the model has to remove the last of the noise, and those terms matter more. Drawing t uniformly from 1…T makes the network learn ε at every noise level.

### DDIM versus DDPM

DDPM's reverse process is a Markov chain of Gaussians. Every step draws a fresh noise sample, the chain is long, and two runs from the same x_T do not match.

DDIM defines a non-Markov update that depends on x_t and on the predicted clean image. With the predicted noise,

```text
x̂_0 = (x_t − √(1 − ᾱ_t) ε_θ) / √ᾱ_t
x_{t-1} = √ᾱ_{t-1} x̂_0 + √(1 − ᾱ_{t-1}) ε̃
```

If η = 0, ε̃ is taken entirely from ε_θ and no new noise is added. The trajectory is deterministic, and a fixed x_T always yields the same image. It is the probability-flow ODE. If η > 0, a controlled amount of noise is added back and the samples diversify.

Because the path is smooth, DDIM can skip most of the time grid. Twenty to fifty steps is often enough, against the hundreds or thousands used by DDPM. The deterministic path is also what makes interpolation and image editing straightforward.

### Classifier-free guidance

Conditional generation can be steered by the gradient of a separate classifier, ∇_x log p_φ(y | x). That means training and running a second model.

Classifier-free guidance uses one network. During training the condition c is dropped on a fraction of the batches (often 10–20%) and replaced by an empty token, so the same weights learn both ε_θ(x_t, t, c) and ε_θ(x_t, t, ∅). At sampling time

```text
ε_CFG = ε_uncond + w (ε_cond − ε_uncond)
```

w = 0 ignores the condition, w = 1 is the ordinary conditional model, and w > 1 amplifies the difference between the two predictions. That difference behaves like the classifier gradient, which is why no classifier is required. Larger w follows the prompt more closely and reduces diversity. Very large w produces artifacts and can collapse the sample. The cost is two forward passes per step, which is still cheaper than a separate classifier. This is the guidance used in practice by models such as Stable Diffusion.

### FashionMNIST experiment

The notebook uses a linear variance schedule with T = 1000, β_1 = 10⁻⁴, and β_2 = 0.02. Then α_t = 1 − β_t and ᾱ_t is the cumulative product, stored as `b_t`, `a_t`, and `a_bar_t`. Early steps add little noise. Late steps arrive at nearly Gaussian noise.

`perturb_input(x0, t)` draws ε ~ N(0, I) and applies the closed form

```text
x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε
```

It returns both x_t and ε. The training loop draws a random t for each image, corrupts it, and trains the U-Net with mean squared error between ε and ε_θ(x_t, t).

Over 10 epochs the loss starts near 1.13 on the first batch, falls to about 0.06–0.10 within that epoch, and then sits around 0.02–0.06. The curve is decreasing and stable. The network has learned to estimate the noise on FashionMNIST.

DDPM sampling starts at x_T ~ N(0, I) and walks t from T − 1 down to 0. At each step the network predicts the noise, the clean image is reconstructed, the reverse mean is applied, and fresh Gaussian noise is added on every step except the last. With 1,000 steps the samples show recognizable clothing (shoes, shirts, coats), with some leftover noise and soft detail.

DDIM sampling uses the same start, but only 20 steps of the grid: t ∈ {999, 949, …, 49, 0}, and the deterministic update with no extra noise. The large jumps make x̂_0 harder to estimate, so the images are blurrier and have more structural artifacts than the 1,000-step DDPM samples.

Timing for 50 images, from the notebook:

| Sampler | Steps | Time |
| --- | --- | --- |
| DDIM | 20 | 404 ms |
| DDPM | 1,000 | 20.3 s |

DDIM is about 50 times faster, which is the reason to use it.

FID against 1,000 validation images, using Inception-v3 features:

| Sampler | FID |
| --- | --- |
| DDPM | 359.97 |
| DDIM | 696.94 |

Lower is closer to the real distribution. The 20-step DDIM run, with this particular subsampled grid and no extra tuning, gives up a large amount of quality for the speed. That matches the images: DDIM is faster, noisier, and statistically farther from FashionMNIST.

## 2. Flow matching on a time series

### The vector field

Flow matching learns a time-dependent vector field v_θ(x, t). Integrating

```text
dx_t / dt = v_θ(x_t, t)
```

from t = 1 to t = 0 carries a Gaussian sample onto the data. The field is the velocity of that flow. If it is learned correctly, the distribution of the endpoint approximates the data distribution.

A diffusion model instead fixes a forward noising process and learns a reverse process, usually a stochastic differential equation, by predicting a score or a reverse-step mean. Flow matching learns the velocity of a deterministic flow directly, in the same family as a continuous normalizing flow or a neural ODE. Sampling is an ODE solve rather than a chain of random steps. The probability-flow ODE of a diffusion model is the point where the two views meet: both are a continuous map from noise to data.

### Linear paths

The usual interpolant is the straight line

```text
x_t = (1 − t) x_0 + t x_1
```

It is easy to differentiate, and if both ends are Gaussian then so is every point on the segment. In high dimension the path is numerically stable, and there is no auxiliary optimization to find the path. It is not optimal in the sense of optimal transport or of the data manifold. A curve could be shorter or could avoid low-density regions. Finding that curve is expensive, so the straight line is a practical compromise.

Because the path is linear, the target velocity and the flow-matching loss have a closed form, usually a mean squared error. Gradients stay smooth and the loss does not swing as it does with a complicated target field. The model also only ever sees distributions that lie between noise and data. A highly curved path can make the target field irregular and the training unstable.

### Score of the Gaussian bridge

The question uses the conditional Gaussian q(x_t | x_τ) = N(x_τ, σ_t² I). Its log-density is

```text
log q(x_t | x_τ) = −(d / 2) log(2π σ_t²) − ||x_t − x_τ||² / (2 σ_t²)
```

Only the second term depends on x_t, and

```text
∇_{x_t} log q(x_t | x_τ) = (x_τ − x_t) / σ_t²
```

The score-matching loss regresses the network onto that target:

```text
L(θ) = E || s_θ(x_t, t) − (x_τ − x_t) / σ_t² ||²
```

Drawing x_t = x_τ + σ_t ε with ε ~ N(0, I) turns the expectation over x_t into an expectation over ε, which is the loss implemented in training.

### SPY log-return experiment

The notebook trains a small MLP vector field on sequences of SPY log returns for 100 epochs. The training MSE moves from about 1.99 at the start to about 1.92–1.93 at the end. The test loss stays nearby, roughly 2.10 down to 2.09. The two curves drop a little in the first epochs and then fluctuate on a plateau. They do not diverge, so there is no clear overfitting. The loss is still high after 100 epochs, which fits a small MLP on noisy financial data: a mild underfitting, not a failure to learn the coarse map from x_0 to x_1.

Synthetic sequences are drawn with `sample_sequences`. The solver starts from a Gaussian at t = 0 and integrates dx/dt = v_θ(x, t) with Euler steps to t = 1. In the normalized space both real and generated paths are small noisy traces around zero. After denormalizing, the amplitude matches and there is no spurious drift. The model reproduces the overall scale of the noise.

Marginal statistics of the flattened returns:

| | Mean | Variance | Volatility |
| --- | --- | --- | --- |
| Real | 0.00048 | 0.0001198 | 0.01095 |
| Generated | 0.00027 | 0.0001328 | 0.01152 |

The generated mean stays near zero and is only slightly below the real mean, so the model does not invent a large bias. Variance and volatility are close, with the generated series a little wider. Histograms and smoothed densities both peak at zero and have light tails. The far tails, the large positive and negative jumps, are where they differ: the model is somewhat conservative about rare moves.

Sliced Wasserstein distance between real and generated paths is about 0.002456. Daily returns are on the order of 10⁻², so a distance of that size is small relative to the data. The paths match in shape and width. What remains is detail and rare events, not the bulk of the distribution.

Autocorrelation, averaged over sequences out to lag 20:

| Metric | Value |
| --- | --- |
| Autocorrelation MSE | 0.00311 |
| Mean absolute ACF difference | 0.04185 |

Both ACFs are 1 at lag 0 and fall quickly, which is the usual near-zero linear correlation of returns. The remaining gap, about 0.04 in absolute difference, means the short-term dependence is only approximate. A fixed-length MLP also does not model volatility clustering or other nonlinear dependence. A recurrent net, a 1-D convolutional net, or a transformer, and a less linear flow, would be the natural next step.

Taken together, the marginal distribution, the volatility, and the short-term autocorrelation are a reasonable match for this assignment. The generated SPY paths are a credible but coarse copy of the series, and the practical part of the question is carried by the notebook.
