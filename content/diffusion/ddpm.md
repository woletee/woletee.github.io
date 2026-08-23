---
title: "DDPM: How Adding Noise Taught Machines to Draw"
date: 2026-08-23
draft: false
weight: 1
math: true
categories: ["Papers", "Diffusion Models"]
tags: ["diffusion", "ddpm", "generative-models", "pytorch", "machine-learning"]
summary: "The 2020 paper that started the diffusion era. Wreck an image with noise, teach a network to guess the noise, then run the wrecking process backwards. Explained with pictures, almost no maths, and code you can run."
description: "A plain-English, picture-first explanation of Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020), with a minimal runnable PyTorch implementation."
cover:
  hidden: true
---

<div class="paper-card">
<dl>
  <dt>Paper</dt>
  <dd><a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Models</a></dd>
  <dt>Authors</dt>
  <dd>Jonathan Ho, Ajay Jain, Pieter Abbeel — UC Berkeley</dd>
  <dt>Published</dt>
  <dd>June 2020 · NeurIPS 2020</dd>
  <dt>Code</dt>
  <dd><a href="https://github.com/hojonathanho/diffusion">hojonathanho/diffusion</a> (original TensorFlow release)</dd>
  <dt>In one sentence</dt>
  <dd>If you slowly destroy an image with noise, a network can learn to undo one small step of that destruction — and running those small steps backwards from pure static produces a brand-new image.</dd>
</dl>
</div>

## The problem nobody had solved cleanly

Generating a realistic image is hard for a reason that is easy to state and
brutal in practice: the space of possible images is unimaginably large, and
almost all of it is garbage. A 256×256 colour photo is roughly 200,000 numbers.
Pick those numbers at random a trillion times and you will never once land on
something that looks like a dog.

So the job of a generative model is to learn where the tiny, tangled, valid
region of that space is, and then produce points inside it.

By 2020 there were three main answers, and each came with a tax:

- **GANs** produced beautiful images, but training them is a fight between two
  networks that can collapse without warning. They also tend to quietly ignore
  parts of the data — you get great dogs and no cats.
- **VAEs** trained reliably but produced blurry output, because they are
  effectively rewarded for hedging.
- **Normalising flows** gave you exact probabilities, but only if you cripple the
  architecture so that every layer stays invertible.

DDPM's contribution is that it sidesteps all three taxes. It trains with plain
mean-squared error, it is about as stable as training an image classifier, it has
no second network to fight with, and the samples are sharp.

The catch — and there is one — is that generating a single image takes a thousand
forward passes. We will get to that.

## The idea, in the form of an analogy

Here is the whole paper as a thought experiment.

Take a photograph. Add a tiny amount of static to it — so little that you can
barely tell. Do it again. And again. A thousand times. By the end you are holding
pure television snow; every trace of the original photo is gone.

Now ask a much smaller question. Not *"can you draw a photo?"* but:

> **"Here is a slightly noisy image. Can you tell me what noise I just added?"**

That question is *easy*. It is a supervised learning problem with a free answer
key, because you were the one who added the noise. You have infinite training data:
take any image, add noise you generated yourself, and you know exactly what the
right answer is.

And here is the payoff. If a network can reliably answer that easy question, you
can start from pure static — which costs nothing, it's just a random number
generator — and repeatedly ask *"what noise is in here?"*, subtract a bit of it,
and ask again. A thousand small subtractions later, an image that never existed
emerges.

{{< diagram src="forward-reverse.svg" caption="**Top:** the forward process. It is a fixed recipe with no learned parameters — you could implement it in five lines. **Bottom:** the reverse process. This is the only part that is a neural network, and it only ever has to undo *one small step*." >}}

<div class="callout">

<span class="callout-title">The key move</span>
Learning to generate an image from nothing is very hard. Learning to make an
image *slightly less noisy* is very easy. DDPM turns the first problem into a
thousand copies of the second one.

</div>

## The forward process is a recipe, not a model

Let's make "add a little noise" precise, because there is one detail that
determines whether the whole thing works.

At each step $t$ you shrink the image slightly toward zero and add a splash of
Gaussian noise:

$$
x_t = \sqrt{1-\beta_t}\; x_{t-1} \;+\; \sqrt{\beta_t}\; \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I)
$$

The numbers $\beta_1, \dots, \beta_T$ are the **noise schedule** — a fixed list
you choose in advance. DDPM uses $T = 1000$ steps with $\beta_t$ rising linearly
from $0.0001$ to $0.02$. Nothing here is learned. It is a recipe.

Why shrink the image by $\sqrt{1-\beta_t}$ instead of just adding noise on top?
Because otherwise the pixel values would grow without bound as you keep adding.
The shrink-and-add pairing is chosen so that the total variance stays fixed at 1
forever. The image doesn't get *louder*, it gets *replaced*.

### The shortcut that makes training possible

If you had to actually run a thousand sequential steps to build one training
example, this would be hopeless. You don't. Because each step is Gaussian, a
thousand of them compose into a single Gaussian, and you can jump straight to any
timestep in one line:

$$
x_t = \sqrt{\bar{\alpha}_t}\; x_0 \;+\; \sqrt{1-\bar{\alpha}_t}\; \epsilon,
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)
$$

Read that as a **crossfade**. The quantity $\sqrt{\bar\alpha_t}$ is how much of
the original image survives, and $\sqrt{1-\bar\alpha_t}$ is how much noise has
taken its place. They always sum in quadrature to one — as one fades in, the
other fades out.

This single formula is what makes training cheap. Want a training example at
$t = 700$? One multiply, one add. No loop.

{{< diagram src="noise-schedule.svg" caption="The crossfade over time under DDPM's linear schedule. By $t \\approx 255$ the image is half signal and half noise; by $t = 600$ only about 16% of the original is left. Notice how much of the schedule is spent in the near-pure-noise region on the right — later papers ([Nichol & Dhariwal](https://arxiv.org/abs/2102.09672)) pointed out that this is mostly wasted, and fixed it with a cosine schedule." >}}

## What the network actually has to do

Now the learned half. The network sees a noisy image $x_t$ and the timestep $t$,
and it outputs a guess of the noise that is in there. That guess is written
$\epsilon_\theta(x_t, t)$.

Three things about this are worth pausing on.

**It predicts the noise, not the image.** You could equally well train it to
output the clean image directly — the two are algebraically interchangeable, since
rearranging the crossfade formula turns one into the other. Ho and colleagues
tried both and found that predicting the noise trains noticeably better. The
intuition people usually give: the noise is a fixed-scale, zero-mean target no
matter what $t$ is, while the clean image has structure that varies enormously
across noise levels. The easier target wins.

**It is told which timestep it is looking at.** One single network handles all
1000 noise levels. An image at $t = 50$ needs a delicate touch; an image at
$t = 900$ needs bold strokes. Without knowing $t$, the network would have to guess
how much damage to undo. So $t$ is converted into a vector — the same sinusoidal
embedding as in Transformers — and injected into every block.

**It is secretly pointing uphill.** There is a deeper reading, which I will
mention once and then leave alone: predicting the noise in $x_t$ is mathematically
the same as estimating the direction in which the image becomes *more probable*
under the data distribution. That direction has a name, the *score*, and it is why
this line of work and the score-matching line of work turned out to be the same
algorithm in different clothes. That connection is [its own
paper](https://arxiv.org/abs/2011.13456), and it is on the reading list.

{{< diagram src="unet.svg" caption="The architecture: a U-Net. The encoder shrinks the picture while widening the channels, letting the middle of the network reason about global layout; the decoder rebuilds full resolution; the skip connections hand fine detail straight across so it does not have to survive the squeeze. Nothing here is diffusion-specific — it is a segmentation architecture from 2015, borrowed wholesale." >}}

## Training: five lines and a random number generator

This is the part that surprises people who expect something exotic. One training
step:

1. Grab a real image $x_0$ from your dataset.
2. Pick a timestep $t$ **uniformly at random** from 1 to 1000.
3. Draw fresh noise $\epsilon \sim \mathcal{N}(0, I)$.
4. Build the noisy image $x_t$ with the crossfade formula.
5. Ask the network to name the noise, and punish it by mean-squared error.

{{< diagram src="training-step.svg" caption="One training step, end to end. Everything on the left is free — the noise is something you generated, so it doubles as the answer key. There is no adversary, no second network, no sampling loop during training." >}}

The entire loss function is this:

$$
L = \mathbb{E}_{x_0, t, \epsilon}
\Big[\;
\big\lVert\, \epsilon - \epsilon_\theta(x_t,\, t) \,\big\rVert^2
\;\Big]
$$

That is it. That is the whole objective. In code:

```python
t = torch.randint(0, T, (batch_size,), device=device)   # random timesteps
eps = torch.randn_like(x0)                              # the answer key
x_t = abar[t].sqrt() * x0 + (1 - abar[t]).sqrt() * eps  # the crossfade
loss = F.mse_loss(model(x_t, t), eps)                   # that's the loss
```

Notice step 2: the timestep is **random**, not sequential. Each gradient step
teaches the network about one randomly chosen noise level. Over a training run it
sees all of them millions of times, and because the weights are shared, skill at
one noise level transfers to its neighbours.

{{< collapse summary="Where the mean-squared error actually comes from (optional)" >}}

The paper does not begin with MSE — it begins with a variational bound, the same
machinery as a VAE, treating the whole noising chain as a 1000-layer latent
variable model. Squeezing that bound down gives a sum of KL divergences, one per
timestep, each comparing two Gaussians.

The KL divergence between two Gaussians with the same variance is just the squared
distance between their means. So each term collapses into a squared error. Grind
through the algebra and you land at a weighted version of the loss above:

$$
L_{\text{weighted}} = \sum_t w_t \,
\big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2
$$

with weights $w_t$ that heavily emphasise the very small timesteps. The paper's
most useful practical finding is that **throwing the weights away and setting
every $w_t = 1$ works better.** Uniform weighting stops the model obsessing over
imperceptible high-frequency detail at $t \approx 0$ and makes it spend capacity
on the noisy steps where the large-scale structure of the image is decided.

So the principled derivation gets you to a weighted MSE, and then an empirical
hack gets you to plain MSE. The simple thing is the thing that works.

{{< /collapse >}}

## Sampling: a thousand small corrections

Training is one step. Generating is the loop.

Start with $x_T$ drawn straight from a random number generator — pure static, no
model involved. Then for $t = T$ down to $1$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}
\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t) \right)
\;+\; \sigma_t z
$$

Ignore the constants and read the shape of it: *take the current image, subtract a
scaled portion of the predicted noise, then add a little fresh noise back.*

{{< diagram src="sampling-loop.svg" caption="The sampling loop. The network is called once per step — a thousand times for one image. This is why DDPM sampling is slow, and why the next three years of research were largely about making this loop shorter." >}}

### Why on earth do we add noise back in?

This is the step that makes everyone stop and re-read, so it is worth answering
properly.

The network's output is a *best guess*, and a best guess is an **average**. At
$t = 900$, when the image is nearly pure static, there are countless different
pictures that could plausibly be hiding underneath. The network cannot know which,
so it predicts something in the middle of all of them — which, averaged over
countless pictures, is a grey blur.

If you followed that average deterministically at every step, you would slide
straight down to the blurriest, most average-looking output the model can produce.
Every sample would look the same, and that same thing would look like fog. This
is exactly the failure mode that makes VAE samples blurry.

Injecting a small amount of fresh randomness at each step breaks the tie. It says:
*of the many images consistent with what we have so far, commit — arbitrarily — a
little further in this direction.* Do that a thousand times and each early
arbitrary nudge gets amplified and made coherent by the steps that follow. The
result is a sharp, specific image rather than the average of all possible images.

<div class="callout warm">

<span class="callout-title">A useful way to hold it</span>
The predicted noise tells you which way to walk. The injected noise decides
*which* valid image you end up at. Remove the second and every walk ends in the
same grey place.

</div>

On the final step ($t = 1$) the noise is skipped, because you want to hand back a
clean image rather than a slightly speckled one.

## What it achieved

DDPM hit an FID of **3.17** on unconditional CIFAR-10, which beat every GAN of the
era on that benchmark and was, at the time, a genuinely startling result for a
non-adversarial model. It also produced convincing 256×256 faces on LSUN.

The number matters less than what it signalled. A simple, stable, MSE-trained
model had just walked past years of adversarial-training research. Everything that
followed — Stable Diffusion, Midjourney, DALL·E 2 and 3, Imagen, Sora — descends
from this paper.

## What it got wrong, and who fixed it

Being honest about the limitations is the most useful part of reading a paper,
because the limitations are the roadmap for everything that came next.

**It is agonisingly slow.** One thousand sequential forward passes per image, and
they cannot be parallelised — each depends on the last. Generating a single
CIFAR-10 image took minutes.
→ Fixed by [DDIM](https://arxiv.org/abs/2010.02502), which reaches comparable
quality in 20–50 steps with *no retraining*, and later by distillation methods
that get to single digits.

**The noise schedule is wasteful.** As the crossfade plot shows, the linear
schedule reaches near-total noise well before $t = 1000$, so a large fraction of
the steps teach the network almost nothing.
→ Fixed by [Improved DDPM](https://arxiv.org/abs/2102.09672) with a cosine
schedule.

**It works on raw pixels.** Every one of those thousand forward passes runs at
full image resolution, which makes high-resolution generation ruinously expensive.
→ Fixed by [Latent Diffusion](https://arxiv.org/abs/2112.10752): compress the
image first, diffuse in the small latent space, decode at the end. This is the
change that put Stable Diffusion on consumer GPUs.

**There is no way to tell it what to draw.** DDPM is purely unconditional. You get
*an* image from the training distribution; you cannot ask for a specific one.
→ Fixed by [classifier guidance](https://arxiv.org/abs/2105.05233) and then, much
more cleanly, [classifier-free guidance](https://arxiv.org/abs/2207.12598).

Every one of these is on the [reading list](/diffusion/).

## A complete implementation you can run

Small enough to read in one sitting, real enough to produce recognisable MNIST
digits in a few minutes on a single GPU. The diffusion logic is about fifteen
lines; the rest is the U-Net.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- the schedule
T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, dim=0)          # \bar{alpha}_t


def add_noise(x0, t, eps):
    """The crossfade: jump straight to timestep t in one operation."""
    a = abar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


# ------------------------------------------------------------------ the model
class TimeEmbedding(nn.Module):
    """Turn the integer t into a vector the network can condition on."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        ang = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class Block(nn.Module):
    """Two convolutions, with the timestep added in between."""

    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.temb = nn.Linear(tdim, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=64, tdim=64):
        super().__init__()
        self.time = TimeEmbedding(tdim)
        td = tdim * 4
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1 = Block(ch, ch, td)
        self.d2 = Block(ch, ch * 2, td)
        self.mid = Block(ch * 2, ch * 2, td)
        self.u2 = Block(ch * 4, ch, td)
        self.u1 = Block(ch * 2, ch, td)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        self.down = nn.AvgPool2d(2)

    def forward(self, x, t):
        te = self.time(t)
        h0 = self.inp(x)
        h1 = self.d1(h0, te)                                      # 28 x 28
        h2 = self.d2(self.down(h1), te)                           # 14 x 14
        m = self.mid(self.down(h2), te)                           #  7 x  7
        u2 = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h2], 1), te)
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2), h1], 1), te)
        return self.out(u1)


# ---------------------------------------------------------------- the training
tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # -> [-1, 1]
)
loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=tf),
    batch_size=128,
    shuffle=True,
    drop_last=True,
)

model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(20):
    for x0, _ in loader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        eps = torch.randn_like(x0)

        loss = F.mse_loss(model(add_noise(x0, t, eps), t), eps)

        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}  loss {loss.item():.4f}")


# ---------------------------------------------------------------- the sampling
@torch.no_grad()
def sample(n=16):
    x = torch.randn(n, 1, 28, 28, device=device)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps_hat = model(x, tt)

        # subtract a slice of the predicted noise
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps_hat) / alphas[t].sqrt()

        # add a pinch back, except on the very last step
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1)


images = sample()
```

Two things to watch while it runs. The loss drops fast and then looks *stuck*
around 0.03–0.05 — that is normal and is not a bug. The loss is averaged over
random timesteps, and the high-noise ones are genuinely close to unpredictable, so
there is an irreducible floor. Judge progress by looking at samples, not at the
number.

## Things that confused me

A few questions I had to work through, in case you have the same ones.

**If the network can predict all the noise in $x_t$, why not subtract all of it
and finish in one step?** You can, and the code is a one-liner. The result is
blurry, for the reason discussed above: at high $t$ the network's prediction is an
average over every image that could be hiding in there. The thousand-step loop
works because each step only asks the network to be right about a *small*
correction, and each step's injected randomness commits to a specific answer that
later steps can build on. (This one-shot estimate is not useless, though — DDIM
uses exactly this quantity, just more carefully.)

**Why random timesteps in training but sequential ones in sampling?** They are
different jobs. Training needs an unbiased sample of the loss across all noise
levels, and random $t$ gives that at one forward pass per example. Sampling needs
an actual trajectory, and the only way down the chain is one rung at a time.

**Does the network memorise 1000 separate denoisers?** No, and that is the point.
Denoising at $t = 500$ and $t = 501$ are nearly identical tasks. Weight sharing
plus the timestep embedding lets the network learn one smooth family of denoisers
indexed by $t$, which is enormously more sample-efficient than 1000 independent
models.

**Is $\epsilon$ during sampling the same $\epsilon$ from training?** No. During
training, $\epsilon$ is a specific noise sample you generated and kept as the
answer key. During sampling there is no answer key — the $z$ added at each step is
fresh randomness whose only job is to keep the walk from collapsing to the mean.

## Where to go next

- The paper itself: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239). It is
  short. With the intuition above in place, the two algorithm boxes on page 4 read
  almost like pseudocode you already know.
- [Lilian Weng's survey](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  is the standard reference for the full derivations.
- Next on this site: **DDIM**, which takes the same trained network and gets
  comparable samples in 50 steps instead of 1000. See the
  [reading list](/diffusion/) for what is coming after that.

*Found an error, or an explanation that did not land? [Tell
me](https://github.com/woletee/woletee.github.io/issues) — I would rather fix it
than leave it wrong.*
