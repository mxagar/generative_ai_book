# Generative Deep Learning: My Notes

These are my notes of the book [Generative Deep Learning, 2nd Edition, by David Foster (O'Reilly)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/).

Table of contents:

- [Generative Deep Learning: My Notes](#generative-deep-learning-my-notes)
  - [Setup](#setup)
  - [Chapter 1: Generative Modeling](#chapter-1-generative-modeling)
  - [Chapter 2: Deep Learning](#chapter-2-deep-learning)
  - [Chapter 3: Variational Autoencoders](#chapter-3-variational-autoencoders)
  - [Chapter 4: Generative Adversarial Networks (GANs)](#chapter-4-generative-adversarial-networks-gans)
  - [Chapter 5: Autoregressive Models](#chapter-5-autoregressive-models)
  - [Chapter 6: Normalizing Flow Models](#chapter-6-normalizing-flow-models)
  - [Chapter 7: Energy-Based Models](#chapter-7-energy-based-models)
  - [Chapter 8: Diffusion Models](#chapter-8-diffusion-models)
  - [Chapter 9: Transformers](#chapter-9-transformers)
  - [Chapter 10: Advanced GANs](#chapter-10-advanced-gans)
  - [Chapter 11: Music Generation](#chapter-11-music-generation)
  - [Chapter 12: World Models](#chapter-12-world-models)
  - [Chapter 13: Multimodal Models](#chapter-13-multimodal-models)
  - [Chapter 14: Conclusion](#chapter-14-conclusion)

## Setup

There is a guide to set up GCP VMs in [`docs/googlecloud.md`](./docs/googlecloud.md); however, I used Google Colab, mainly with T4 GPUs.

The necessary installation/downloads are locally noted in each notebook.

## Chapter 1: Generative Modeling

- Discriminative vs. generative models
  - Discriminative models: p(y|X), probability of a label y given some observation X.
    - Used to be the ones applied in practice, easier to implement.
  - Generative models: p(X), probability of an observation X. Sampling from this distribution allows to generate new observations!
    - Were though to be in-applicable; but not anymore!
- Representation learning aims to learn compressed (lower-dimensional) latent representations that can be later expanded to high-dimensional spaces.
  - Latent representations are easier to handle.
- Symbols, concepts
  - Sample space: space of all X observations
  - Probability density function: p(X), probability of an observation X; we distinguish:
    - p_data
    - p_model: estimates p_data
  - Parametric modeling: parametrized p
  - Likelihood: plausibility of the parameter values theta given some observation X
  - Maximum likelihood estimation: technique that allows to estimate the parameter set theta
- Type of generative models
  - Explicit p(), density
    - Approximate: often latent variables used
      - VAEs
      - Energy-based
      - Diffusion
    - Tractable: they place constraints for easier calculation, e.g., sequence order
      - Autoregressive
      - Normalizing Flow
  - Implicit p(), density: they have no formal/explicit construct of p
    - GANs

## Chapter 2: Deep Learning

- Deep Learning = Deep Neural Networks, i.e., ANNs with more than 1 hidden layer.
- Structured vs. unstructured data:
  - Structured data is tabular data, with features already extracted.
  - Unstructured data have high dimensionality and features are not extracted: images, text, audio, etc.
- DL works very nicely with unstructured data and is able to extract automatically features from it!
- Concepts of a ANN
  - Layers, units, weights, activation functions
  - MLP = Multi-Layer Perceptron = fully connected layer.
  - Forward pass
  - Training: backpropagation
  - Evaluation, inference
- Example: MLP for CIFAR-10 classification using Keras (32x32x3 images and 10 classes)
  - Keras: Sequential vs. Functional API
  - Steps:
    - Normalize image tensors
    - One-hot-encode labels
    - Train and test splits
    - Model definition
      - Flattening of tensors
      - Layers: Dense, Activation (ReLU, LeakyReLU, Softmax)
      - Batch size defined online
      - Counting parameters: mind the bias!
    - Loss functions
      - Regression: Mean Square Error, Mean Absolute Error
      - Classification: Binary cross-entropy, Categorical Cross-entropy
    - Optimizers: usually, only learning rate is modified
      - Adam
      - RMSProp
    - Training
      - Random initialization of weights
      - Epochs
      - Metrics: accuracy, loss
    - Evaluation: Compute metrics with test set (unseen)
- Example: CNN for CIFAR-10 classification using Keras (32x32x3 images and 10 classes)
  - CNNs: 
    - in a fully connected layer, image data is flattened, so we loose spatial information
    - in a CNN the spatial information is preserved, plus, we have less parameters!
  - Convolution, kernel, filter
    - Stride, padding, depth / channels
    - Note: if we apply to (32,32,3) a 10-channel filter of size (4,4):
      - Output image has 10 channels
      - We have 10 filters of size 4x4x3, so (4x4x3 + 1(bias))x10 = 490 parameters
  - Batch normalization
    - Exploding gradient
    - Normalization of batches
    - Scale, shift, moving average, moving variance
    - Smoother gradients
  - Dropout: random de-activation of units only during training to achieve regularization
  - A CNN will usually achieve a much better accuracy on images than a MLP; it also has usually less parameters!

## Chapter 3: Variational Autoencoders

## Chapter 4: Generative Adversarial Networks (GANs)

## Chapter 5: Autoregressive Models

## Chapter 6: Normalizing Flow Models

## Chapter 7: Energy-Based Models

## Chapter 8: Diffusion Models

## Chapter 9: Transformers

## Chapter 10: Advanced GANs

## Chapter 11: Music Generation

## Chapter 12: World Models

## Chapter 13: Multimodal Models

## Chapter 14: Conclusion


