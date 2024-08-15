# Generative Deep Learning: My Notes

These are my notes of the book [Generative Deep Learning, 2nd Edition, by David Foster (O'Reilly)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/).

Table of contents:

- [Generative Deep Learning: My Notes](#generative-deep-learning-my-notes)
  - [Setup](#setup)
  - [Chapter 1: Generative Modeling](#chapter-1-generative-modeling)
    - [Key points](#key-points)
  - [Chapter 2: Deep Learning](#chapter-2-deep-learning)
    - [Key points](#key-points-1)
    - [Notebooks](#notebooks)
  - [Chapter 3: Variational Autoencoders](#chapter-3-variational-autoencoders)
    - [Key points](#key-points-2)
    - [Notebooks](#notebooks-1)
    - [List of papers](#list-of-papers)
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

See also:

- [mxagar/tool_guides/hugging_face](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [mxagar/generative_ai_udacity](https://github.com/mxagar/generative_ai_udacity)
- [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs)
- [mxagar/nlp_guide](https://github.com/mxagar/nlp_guide)

## Setup

There is a guide to set up GCP VMs in [`docs/googlecloud.md`](./docs/googlecloud.md); however, I used Google Colab, mainly with T4 GPUs.

The necessary installation/downloads are locally noted in each notebook.

In each notebook opened in Colab, we need to run the following commands in the beginning:

```python
# Run these lines in Colab
!git clone https://github.com/mxagar/generative_ai_book.git
!mkdir notebooks
!mkdir chekpoint
!mkdir outputs
!mkdir models
!mv generative_ai_book/notebooks/utils.py ./notebooks/utils.py
```

## Chapter 1: Generative Modeling

### Key points

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

### Key points

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

### Notebooks

- [`mlp.ipynb`](./notebooks/02_deeplearning/01_mlp/mlp.ipynb)
- [`cnn.ipynb`](./notebooks/02_deeplearning/02_cnn/cnn.ipynb)

## Chapter 3: Variational Autoencoders

In this chapter **Autoencoders** and **Variational Autoencoders** are introduced; the latter improve the properties of the latent space that compresses the data.

### Key points

- Autoencoders have an encoder-decoder architecture with a bottleneck in the middle, which contains the latent/embedding vectors.
  - They are usually trained to compress and expand the input data, by optimizing for minimal reconstruction error.
  - Then, we can use de decoder only to create latent/embedding representations, or the decoder only to generate new unseen data points.
  - Applications: vectorization, anomaly detection, de-noising.
- Example of Autoencoder with FashionMNIST dataset: 28x28 grayscale images
  - Latent vector in the example, z, is 2D; in practice, we'll have more than 2 dimensions, although having too large dimensionalities leads to problems.
  - Encoder: `Conv2D x3`
  - Decoder: equivalent to the encoder, but with expansion, for images using `Conv2DTranspose (x3)`
    - Convolutional transpose layers: the filter is put on a target pixel and multiplied by all cells to create a patch with the filter size; depending on the stride, these patches can overlap, then, cell values are summed.
    - The output of the last `Conv2DTranspose` is passed to a final `Conv2D` layer.
  - Loss function: can be RMSE of complete image of pixel-wise binary cross-entropy
    - RMSE: errors symmetrically/homogeneously penalized.
    - Binary cross-entropy: extreme errors more heavily penalized; blurrier images.

### Notebooks

- []()
- []()

### List of papers

- VAE (Kingma, 2013): [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

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


