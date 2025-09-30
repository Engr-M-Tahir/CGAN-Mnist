# Conditional GAN (CGAN) from Scratch on MNIST

Project Overview

This project implements a Conditional Generative Adversarial Network (CGAN) from scratch using PyTorch, trained on the MNIST handwritten digits dataset. Unlike traditional GANs, CGANs allow generation of data conditioned on class labels, enabling control over the digit being generated (0–9).
The project is built entirely from scratch without relying on pre-built GAN modules, showcasing a deep understanding of GAN architecture, training stability, and conditional image generation.

Key Features

* Conditional GAN Implementation — Both Generator and Discriminator are conditioned on digit labels.

* Label Embeddings — One-hot encoded class labels embedded into both Generator and Discriminator.

* Image Generation Control — Ability to generate a specific digit by providing its label.

* Training from Scratch — Fully custom architecture written in PyTorch.

* Evaluation — Visualizations of generated digits over training epochs.


Dataset

* MNIST Handwritten Digits (60,000 training images, 10,000 test images).

* Each image: 28×28 grayscale.

* Labels: 10 classes (digits 0–9).
