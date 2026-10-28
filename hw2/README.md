# Homework 2 — Normalizing flows and CycleGAN

Notebooks:

- [maf.ipynb](maf.ipynb) — masked autoregressive flow, sampling, and anomaly detection on MVTec
- [cyclegan.ipynb](cyclegan.ipynb) — unpaired horse ↔ zebra translation

## 1. Masked autoregressive flow

### Model

A MADE network is an autoregressive network with masked weights. For each input dimension it predicts the parameters of that dimension, and the masks force dimension i to depend only on dimensions before it. The masks are built in `create_masks` and applied by `MaskedLinear` layers.

The MADE used here has three masked linear layers:

- `fc1` maps the input to the hidden layer
- `fc2` maps hidden to hidden
- `fc3` maps hidden to the output, of size `2 * input_dim`

`reset_masks()` builds those masks and writes them onto the layers.

Each `MAFBlock` runs MADE and splits the output in half:

- `input_dim` is the dimension of x
- `hidden_dim` is the width of the MADE hidden layers
- the first half of the output is the shift
- the second half is `log_scale`

`log_scale` is clamped. Without that clamp, `exp(-log_scale)` can overflow and the loss becomes NaN or Inf.

The forward pass is used in training. Sampling runs the inverse, from the base noise z back to x.

### Practical setup

Training was done on Colab. Images were resized to 64×64×3 to keep the computation smaller, and the hidden width was set to 1024.

The flow was trained for 150 epochs. Five generated images are shown in the notebook. At 64×64 the samples are soft and low in detail.

Total training time is longer than the time spent generating those five images, because training covers 150 epochs. Per epoch, training is much cheaper than generating one image. Autoregressive sampling is slow: each pixel depends on every pixel before it, so the pixels cannot be drawn in parallel. The inverse model has the opposite cost profile. Its training pass is sequential and slow, and its sampling pass is parallel and fast.

### Anomaly detection

Accuracy is a poor metric here. Normal images dominate the set (for example 99% normal and 1% anomalous), so a classifier that always says "normal" can score 99% and still be useless. Missing an anomaly is also usually more costly than flagging a normal image. The point of the task is to catch the rare cases, not to maximize overall accuracy.

More suitable scores:

| Metric | What it measures |
| --- | --- |
| AUROC | How well the score separates normal from anomalous images. 0.5 is chance, 1.0 is perfect. |
| AUPRC | The same idea on the precision–recall curve. It is the better choice on an imbalanced set. |
| F1 | The harmonic mean of precision and recall, when both kinds of error matter. |
| Precision | The fraction of images flagged as anomalous that really are anomalous. |
| Recall | The fraction of true anomalies that the model catches. |

In a normalizing flow the anomaly score is the negative log-likelihood. A normal image sits near the learned density, so its likelihood is high and its score is low. An anomalous image has a low likelihood and a high score. The flow is trained only on normal images, so anything that does not fit that density is treated as anomalous.

At test time, log p(x) is evaluated for each image and the score is NLL = −log p(x). A threshold on that score splits normal from anomalous. The threshold can be set from an allowed false-positive rate, or from a validation set.

The notebook runs this on the MVTec capsule class and plots the ROC and precision–recall curves.

A VAE cannot compute an exact likelihood, but it can still be used for the same task. The reconstruction error replaces the negative log-likelihood as the anomaly score.

## 2. CycleGAN

### Why unpaired translation needs a cycle

Paired supervision is rarely available. Matching two photos of the same scene in light, angle, motion, and setting is often impossible. CycleGAN trains on two separate collections instead of matched pairs.

The objective has four terms. The first two are the usual adversarial losses, one for each direction: they score the generators and the discriminators. The other two are cycle-consistency losses. They measure how well an image survives a round trip from one domain to the other and back, and they are what ties the two domains together.

If the cycle term is removed, nothing forces the generator to keep the content of the input. It can emit any image the discriminator accepts. Training becomes unstable, the mapping is no longer controllable, and the model is free to collapse into a meaningless translation. The cycle term is what blocks those mappings.

If the cycle weight λ is too large, the generators care more about reconstructing the input than about changing its style. The maps drift toward the identity. The output keeps the structure of the input, but it no longer looks like a real sample of the target domain.

The identity loss applies when the input is already in the target domain. The generator should then leave it alone: no extra change in brightness, color, texture, or structure. For a horse-to-zebra map, a real zebra passed to that generator should come back as the same zebra.

### Two generators, not one

The two maps are not the same function. G goes from B to A and F goes from A to B, and the two domains are not symmetric, so one shared generator is a poor fit.

- Each direction may need its own depth or receptive field. Separate generators can be tuned separately.
- One network asked to map both ways receives conflicting gradients and tends to oscillate.
- The domains can differ in color, texture, and structure. One set of weights does not model both styles well at once.

### Generator and discriminator design

pix2pix translates paired images with a U-Net. Skip connections carry spatial detail from the encoder to the decoder, which is what a pixel-aligned translation needs (semantic map to photo, sketch to photo).

CycleGAN translates unpaired images with a ResNet generator: an encoder, a stack of residual blocks, and a decoder. It does not use skip connections. The residual stack is aimed at style and texture, not at a pixel-to-pixel copy.

The discriminator is a PatchGAN. Style lives in local texture, and a patch discriminator judges those patches instead of the whole image. That improves small regions, uses fewer parameters, trains faster, and does not depend on one fixed image size.

### Replay buffer

If the discriminator only ever sees the generator's latest output, the two networks chase each other: the generator jumps, the discriminator catches up, the generator jumps again. That oscillation is a common route to mode collapse. A buffer of older generated images steadies the discriminator, reduces collapse, and improves the final samples.

### What CycleGAN does not do

CycleGAN changes style, not geometry. It can change texture, color, and pattern. It does not reliably change shape, size, pose, or object class. Horse to zebra works because the change is mostly texture. Horse to bird, or a large change of pose, does not.

The cycle constraint is why. F(G(a)) has to come back to a. A large geometric change is hard to undo, so the cycle loss penalizes it, and the generator learns to avoid it.

### Practical run

The notebook loads random images from both domains, builds the two generators and two discriminators, and trains with the adversarial, cycle, and identity terms. The hyperparameters were chosen to fit the available GPU. Intermediate samples, final horse-to-zebra and zebra-to-horse grids, and the loss curves are in the notebook.

The loss curves behave like the usual GAN game: the generator and discriminator losses keep pushing on each other rather than falling smoothly to zero.
