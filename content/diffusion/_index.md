---
title: "Diffusion Models"
description: "A growing library of diffusion model papers, each rewritten as a plain-English explanation with pictures and runnable code. Start at the beginning and read forward."
summary: "A growing library of diffusion model papers, each rewritten as a plain-English explanation with pictures and runnable code."
---

Almost every image, video, and audio generator you have used in the last few years
runs on the same idea: **learn to turn static into a picture, one small step at a
time.** That idea is barely five years old, and the papers that built it are
readable — once someone strips away the notation and tells you what is actually
going on.

That is what this section is for. One paper at a time, I rewrite the paper the way
I wish it had been explained to me:

- **the question it was trying to answer**, and why the previous answers were unsatisfying
- **the one idea** that makes the paper work, in words before symbols
- **pictures**, because most of these papers are much easier to see than to read
- **code you can run**, small enough to read in one sitting
- **what it got wrong**, and which paper fixed it

You do not need a measure-theory background. If you are comfortable with
"a neural network takes an input and produces an output" and you can squint at a
normal distribution without flinching, you have enough.

## Start here

If you read nothing else, read the first one. Every other paper on this list is a
modification of it.

<ul class="roadmap">
  <li class="done">
    <span class="rm-title"><a href="/diffusion/ddpm/">Denoising Diffusion Probabilistic Models</a></span>
    <span class="rm-year">Ho, Jain &amp; Abbeel · 2020</span>
    <span class="rm-status">Explained</span>
    <span class="rm-note">The paper that made diffusion work. Destroy an image with noise, train a network to undo one step of the damage, then run it backwards from pure static.</span>
  </li>
</ul>

## The rest of the map

These are the papers I am working through next, roughly in the order that makes
them easiest to understand. Each builds on the one above it.

<ul class="roadmap">
  <li>
    <span class="rm-title">Denoising Diffusion Implicit Models (DDIM)</span>
    <span class="rm-year">Song, Meng &amp; Ermon · 2020</span>
    <span class="rm-status">Next up</span>
    <span class="rm-note">Sampling in 50 steps instead of 1000, without retraining anything. Also the paper that makes diffusion models deterministic enough to interpolate between images.</span>
  </li>
  <li>
    <span class="rm-title">Improved Denoising Diffusion Probabilistic Models</span>
    <span class="rm-year">Nichol &amp; Dhariwal · 2021</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">The engineering pass: the cosine noise schedule, learning the variance instead of fixing it, and why the original schedule wastes steps.</span>
  </li>
  <li>
    <span class="rm-title">Score-Based Generative Modeling through SDEs</span>
    <span class="rm-year">Song et al. · 2021</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">The view from above: DDPM and score matching are the same algorithm, and both are a differential equation you can solve with any solver you like.</span>
  </li>
  <li>
    <span class="rm-title">Diffusion Models Beat GANs on Image Synthesis</span>
    <span class="rm-year">Dhariwal &amp; Nichol · 2021</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">Classifier guidance, and the moment diffusion overtook GANs on ImageNet.</span>
  </li>
  <li>
    <span class="rm-title">Classifier-Free Diffusion Guidance</span>
    <span class="rm-year">Ho &amp; Salimans · 2022</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">Two forward passes and a subtraction, and suddenly prompts are obeyed. The single most consequential trick in text-to-image generation.</span>
  </li>
  <li>
    <span class="rm-title">High-Resolution Image Synthesis with Latent Diffusion Models</span>
    <span class="rm-year">Rombach et al. · 2022</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">Stable Diffusion. Do the diffusion in a compressed latent space instead of on pixels, and the whole thing fits on a consumer GPU.</span>
  </li>
  <li>
    <span class="rm-title">Scalable Diffusion Models with Transformers (DiT)</span>
    <span class="rm-year">Peebles &amp; Xie · 2022</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">Replace the U-Net with a transformer and the scaling laws come along for free. The backbone behind Sora and most modern video models.</span>
  </li>
  <li>
    <span class="rm-title">Flow Matching / Rectified Flow</span>
    <span class="rm-year">Lipman et al. · Liu et al. · 2022</span>
    <span class="rm-status">Planned</span>
    <span class="rm-note">A cleaner reformulation that quietly replaced diffusion training in most 2024-and-later systems. Straight paths instead of a noisy random walk.</span>
  </li>
</ul>

<div class="callout">
<span class="callout-title">A note on how to read these</span>
Every page follows the same shape, so you can skim the parts you already know.
There is a summary box at the top with the links and the one-sentence
contribution, then intuition, then pictures, then code. Optional derivations are
tucked inside collapsible sections — open them if you want them, skip them if you
do not. Nothing later depends on them.
</div>

Missing a paper you think belongs here, or spotted something I got wrong?
[Open an issue](https://github.com/woletee/woletee.github.io/issues) — corrections
are genuinely welcome.
