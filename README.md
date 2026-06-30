# Introduction
This project provides a fully from scratch implementation of neural networks and backpropagation training. It is based on first principles and fully derived [on my website](https://philsfun.com/neural/index.html). The provided model (mnist-200-100.zip) was trained on MNIST data and reached an accuracy of 98.5%. For a live demo, please see the website.

<p align="center">
<img width="600" alt="accuracy" src="https://github.com/user-attachments/assets/34b7c058-421d-4c2b-a366-1cb93da44d09" />
</p>

# Overview
Given an $n$-layer perceptron $f(x^{(0)};\vartheta) := x^{(n)}$ defined by the forward-pass equation

$$x^{(k)} := \sigma\Big(W^{(k)}x^{(k-1)}+b^{(k)}\Big)$$

and input-output pairs $(x,t) \sim P_\text{data}$ the problem is to find weights $W^{(k)}$ and biases $b^{(k)}$ such that

$$\min_\vartheta \underset{{P_\text{data}}}{\mathbb{E}} \Big[\mathcal{L}(f(x;\vartheta),t)\Big]$$

where $\vartheta := \Big\\{W^{(k)},b^{(k)}|k\in[n]\Big\\}$ and $\mathcal{L}$ is a loss function. This can be solved using stochastic gradient descent. Please refer to the write-up on my website if you want to learn more.

# Features
Lets you define and train your own multilayer perceptron with any topology. Allows for any activation / loss function you want. Allows for any SGD optimizer you want (default: AdamW). Allows for any learning schedule with or without warmup. Trains using backpropagation and automatically logs accuracy.
