# Generative Deep Learning: My Notes

These are my notes of the book [Generative Deep Learning, 2nd Edition, by David Foster (O'Reilly)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/).

Table of contents:

- [Generative Deep Learning: My Notes](#generative-deep-learning-my-notes)
  - [Setup](#setup)
    - [Google Colab Setup](#google-colab-setup)
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
    - [Key points](#key-points-3)
    - [Notebooks](#notebooks-2)
    - [List of papers](#list-of-papers-1)
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

### Google Colab Setup

In each notebook opened in Colab, we need to run the following commands in the beginning:

```python
# Run these lines in Colab
!git clone https://github.com/mxagar/generative_ai_book.git
!mkdir notebooks
!mkdir chekpoint
!mkdir data
!mkdir output
!mkdir models
!mv generative_ai_book/notebooks/utils.py ./notebooks/utils.py
!mv generative_ai_book/notebooks/03_vae/03_vae_faces/vae_utils.py ./notebooks/vae_utils.py
!pip install python-dotenv
```

Also, we can install kaggle properly and download the `kaggle.json` to `~/.kaggle`; alternatively, we can set the following environment variables in `.env`:

```bash
# You can create an account for free and get an API token as follows:
# kaggle.com > account Settings > API > Create new API token
KAGGLE_USERNAME=xxx
KAGGLE_KEY=xxx
```

Additionally, we need to define the notebook-specific variables `KAGGLE_DATASET_USER` and `KAGGLE_DATASET`, which refer to the datasets we'd like to download, if necessary (see `./scripts/download.sh`).

```python
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set an environment variable
# Get values from ./scripts/download.sh
# E.g., celebrity faces dataset: jessicali9530, celeba-dataset
os.environ['KAGGLE_DATASET_USER'] = 'jessicali9530'
os.environ['KAGGLE_DATASET'] = 'celeba-dataset'
```

And finally, we need to download the proper dataset, if required:

```python
import os
os.system("cd data/ && kaggle datasets download -d $KAGGLE_DATASET_USER/$KAGGLE_DATASET")
!unzip -q -o data/$KAGGLE_DATASET.zip -d data/$KAGGLE_DATASET
!rm data/$KAGGLE_DATASET.zip
```

Finally, we'll probably need to correct the paths to the dataset from `/app/data/...` to `./data/...`.

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
  - Since the model is defined as two sequential encoder and decoder sub-models, we can use
    - the encoder to create image embeddings,
    - the decoder to generate new images from random embedding vectors,
    - all the model to reconstruct the image, i.e., for instance, to de-noise.
  - Observations of the latent space:
    - Autoencoders have latent spaces with discontinuities and somewhat arbitrary, i.e., not really uniformly distributed.
    - Thus, it's difficult to interpolate new images correctly on those spaces.
    - On the contrary, Variational Autoencoders don't have those issues! Their latent space is more structured and uniformly distributed, so we can interpolate.
  - Same Example with Variatioanal Autoencoders
    - Two major changes:
      - We predict regions/patches in the latent space instead of points, i.e., we predict the mean and standard distribution of a distribution for each observation. In practice: mean and log-variance.
      - We require those regions to have a location close to the center and a given width; effect: better, more distributed latent space.
    - For each latent dimension, we obtain two values: `z_mean, z_log_var`.
    - The different variances of each dimension in latent space are assumed to be independent from each other, i.e., we assume an isotropic, diagonal covariance matrix.
    - To run a complete forward pass of an observation we need to sample in the latent distribution. Thus, there is a `Sampling` layer.
      - Reparametrization trick: to enable backpropagation, only one `epsilon` value is randomly generated in a standard distribution and the the mean and variance are adjusted.
    - The loss function has two components:
      - Reconstruction loss, as in the regular autoencoder.
      - The Kullback-Leibler (KL) divergence: how much a probability distribution differs from another; in our case, how much it differs from the standard distribution. The objective would be `z_mean = z_log_var = 0`.
      - Beta-VAEs use a `beta` hyperparameter to weight between the two losses.
    - The `training_step` is implemented with the `GradientTape` context, which automatically computes the gradients of the operations.
  - Example: (Beta-) Variational Autoencoder with CelebA dataset
    - 200k color images, resized to 32 x 32 pixels.
    - Labels are present for later, not used during training: glasses, smiling, hat, mustache, etc.
    - Latent space: 200 dimensions.
    - Batch normalization used, but apart from that, architecture similar to previous one.
    - Latent space arithmetics:
      - Morphing between images is a linear operation with only one varied parameter.
      - Since we have labels (e.g., smiling), we can compute these feature vectors and add substract them to other images:
        - `smile_vector = avg(images_smiling) - avg(images_all)`
        - if we add `smile_vector` to a non-smiling vector, with different scaling factors, the image starts smiling!


### Notebooks

- [`autoencoder.ipynb`](./notebooks/03_vae/01_autoencoder/autoencoder.ipynb)
- [`vae_fashion.ipynb`](./notebooks/03_vae/02_vae_fashion/vae_fashion.ipynb)
- [`vae_faces.ipynb`](./notebooks/03_vae/03_vae_faces/vae_faces.ipynb)

### List of papers

- VAE (Kingma, 2013): [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

## Chapter 4: Generative Adversarial Networks (GANs)

### Key points

- GAN = Generative Adversarial Networks (by Goodfellow)
  - A battle between two submodels:
    - Generator: tries to generate realistic images as if the belonged to a dataset distribution, starting with a noise seed/latent vector.
    - Discriminator: tries to determine whether an image is real or fake (i.e., generated by the Generator); binary output.
    - Usually, D and G have mirrored architectures.
  - DCGAN: Deep Convolutional GAN; convolutions and transpose convolutions are used.
    - Discriminator: compression using convolution, batch normalization, leaky ReLU and dropout. Channels increase as we go deeper.
    - Generator: expansion using transpose convolutions instead of convolutions. Channels decrease as we go deeper.
      - It has been shown that the transpose convolutions produce some checkerboard patterns/artifacts, although they are actively used; an alternative is to use upsampling followed by a convolution.
- DCGAN Example: Bricks dataset, 40k photographs of 50 LEGO bricks, taken from all angles; 64 x 64 pixels.
  - Images are usually re-scaled to `[-1,1]` when working with GANs (`tanh` output).
  - A custom `DCGAN` model class is created with a custom `train_step()`:
    - First, we train the Discriminator:
      - We take real images: `x`.
      - We create fake images with the Generator from `z` noise vectors: `x = G(z)`.
      - We pass the real and fake batches to the Discriminator with the correct label: `D(x), D(G(z))`.
      - Error is backpropagated to **update only Discriminator weights**.
    - Then, we train the Generator:
      - We create fake images with the Generator from `z` noise vectors: `x = G(z)`.
      - We pass the fake batches to the Discriminator: `D(G(z))`.
      - Error is backpropagated to **update only Generator weights**.
    - Both phases are alternated
    - D and G are fighting for dominance, thus, the learning curves and the training might seem unstable.
      - If the D is dominating, we can reduce its weight by introducing some incorrect labels, i.e., noise. This is called **label smoothing**. It generally improves the stability of the training.
  - GAN Training tips: GANs are notoriously difficult to train.
    - Often, the Discriminator becomes too strong, it dominates; if that's the case:
      - Increase Dropout rate in D.
      - Reduce learning rate of D.
      - Reduce convolutional filters in D.
      - Add noise to labels when training D: flip labels.
    - Sometimes, the Generator can overpower the Discriminator
      - The Generator might find a set of images which always fool the Discriminator: mode collapse.
      - Solution: use the opposite suggestions as explained in the previous point, i.e., strengthen the D.
    - Uninformative loss: The G is graded against the current D, but the D is improving; thus, the loss value of 2 points in time is not really comparable.
    - Hyperparameters: GANs are very sensitive to small changes.
    - An improved implementation which alleviates all those issues is the **Wasserstein GAN**.
- Wasserstein GAN with Gradient Penalty (WGAN-GP)
  - This approach introduces a new loss, which alleviates all the aforementioned issues.
  - The Wasserstein loss is similar to the regular binary cross-entropy loss, but we remove the `log`.
  - In addition, we need to add/enforce the Lipschitz constraint, i.e., the absolute value of the gradient must be at most 1 everywhere; this is achieved with **Gradient Penalties**.
  - As a result, we don't need to care about any dominance issues, all learning curves converge.
- WGAN-GP Example: CelebA dataset, face generation
  - As compared to VAEs, GANs produce crisper images, better defined; however, GANs are more difficult to train and require longer training times.
- Conditional GAN (CGAN)
  - 

### Notebooks

- [`dcgan.ipynb`](./notebooks/04_gan/01_dcgan/dcgan.ipynb)
- [`wgan_gp.ipynb`](./notebooks/04_gan/02_wgan_gp/wgan_gp.ipynb)
- [`cgan.ipynb`](./notebooks/04_gan/03_cgan/cgan.ipynb)

### List of papers

- GAN (Goodfellow et al., 2014) [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- DCGAN, Deep Convolutional GAN (Radford et al., 2015): [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Wasserstein GAN (Arjovsky et al., 2017): [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- CGAB, Conditional GAN (Mirza et al., 2014): [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

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


