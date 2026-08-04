---
title: "Understanding Transformers From a Research Paper"
date: 2026-08-04
draft: false
categories: ["Papers"]
tags: ["machine-learning", "transformers", "python"]
---

## Paper

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## The problem

Explain the problem using your own words.

## Main idea

Describe the important concepts, diagrams, and equations.

## Python implementation

```python
import numpy as np

def softmax(values):
    exp_values = np.exp(values - np.max(values))
    return exp_values / exp_values.sum()

print(softmax(np.array([1.0, 2.0, 3.0])))
```

## What I learned

Summarize the practical lessons and link to your complete GitHub project.