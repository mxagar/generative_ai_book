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
    - [Key points](#key-points-4)
      - [Text Generation with LSTMs](#text-generation-with-lstms)
      - [Image generation with PixelCNN](#image-generation-with-pixelcnn)
    - [Notebooks](#notebooks-3)
    - [List of papers](#list-of-papers-2)
  - [Chapter 6: Normalizing Flow Models](#chapter-6-normalizing-flow-models)
    - [Key points](#key-points-5)
    - [Notebooks](#notebooks-4)
    - [List of papers](#list-of-papers-3)
  - [Chapter 7: Energy-Based Models](#chapter-7-energy-based-models)
    - [Key points](#key-points-6)
    - [Notebook](#notebook)
    - [List of papers](#list-of-papers-4)
  - [Chapter 8: Diffusion Models](#chapter-8-diffusion-models)
    - [Key points](#key-points-7)
      - [More details](#more-details)
    - [Notebooks](#notebooks-5)
    - [List of papers and links](#list-of-papers-and-links)
  - [Chapter 9: Transformers](#chapter-9-transformers)
    - [Key points](#key-points-8)
    - [Notebooks](#notebooks-6)
    - [List of papers and links](#list-of-papers-and-links-1)
  - [Chapter 10: Advanced GANs](#chapter-10-advanced-gans)
    - [Key points](#key-points-9)
    - [Notebooks](#notebooks-7)
    - [List of papers and links](#list-of-papers-and-links-2)
  - [Chapter 11: Music Generation](#chapter-11-music-generation)
  - [Chapter 12: World Models](#chapter-12-world-models)
  - [Chapter 13: Multimodal Models](#chapter-13-multimodal-models)
  - [Chapter 14: Conclusion](#chapter-14-conclusion)

See also:

- [mxagar/tool_guides/hugging_face](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [mxagar/generative_ai_udacity](https://github.com/mxagar/generative_ai_udacity)
- [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs)
- [mxagar/nlp_guide](https://github.com/mxagar/nlp_guide)
- [mxagar/computer_vision_udacity/CVND_Advanced_CV_and_DL.md](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md)
- [mxagar/deep_learning_udacity/DLND_RNNs.md](https://github.com/mxagar/deep_learning_udacity/blob/main/04_RNN/DLND_RNNs.md)

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

# Download dataset
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
    - Discriminator (aka. Critic): tries to determine whether an image is real or fake (i.e., generated by the Generator); binary output.
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
- Wasserstein GAN with Gradient Penalty (WGAN-GP): It solves the issues of a regular GAN / DCGAN
  - This approach introduces a new loss, which alleviates all the aforementioned issues.
  - The Wasserstein loss is similar to the regular binary cross-entropy loss, but we remove the `log`.
  - In addition, we need to add/enforce the Lipschitz constraint, i.e., the absolute value of the gradient must be at most 1 everywhere; this is achieved with **Gradient Penalties**.
  - As a result, we don't need to care about any dominance issues, all learning curves converge.
- WGAN-GP Example: CelebA dataset, face generation
  - As compared to VAEs, GANs produce crisper images, better defined; however, GANs are more difficult to train and require longer training times.
- Conditional GAN (CGAN): we can generate blonde-hair/brown-hair faces, if we have the labels
  - If we pass the labels of image attributes, we can generate new images with given attributes.
  - Generator: we concatenate a one-hot encoded vector of the attributes.
  - Discriminator / Critic: extra channels created with the one-hot encoded labels.
  - We can use as many labels as we want, but everything needs to be coherent: number of concatenated values or channels, the corresponding images, etc.
  - The resulting GAN is able to organize latent points in such a way that the labeled attributes are decoupled! So we can create the same face but with blonde/brown hair.

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

This chapter contains two autoregressive models:

- Text generation with LSTMs.
- Image generation with PixelCNN.

**Autoregressive** models treat the gerenation as a **sequential** process: instead of using a latent variable, they use they condition the current prediction on the previous values in the sequence, which can be even previously predicted values.

### Key points

#### Text Generation with LSTMs

- LSTMs can be used not only for generating text, but for processing any sequential data.
- LSTM cells are basically a layer which performs several recurrent operations
  - Recurrent becuse they use their previous outputs, aka. hidden states.
  - In contrast to naive recurrent layers
    - they do not suffer from the vanishing gradient issue, even with sequences of hundrends of steps
    - they keep track of two hidden states: cell state `C` (long-term memory, cell's belief) and hidden state `h` (short-term memory, actual output).
    - they internally have 4 gates in which they apply operations with feed-forward matrices to *remember* and *forget* input/state information
- Dataset: 20k recipe texts with metadata (nutritional information)
  - Text data is different from images:
    - Text is composed of discrete chunks.
    - Tt's straightforward to interpolate from a blue pixel to a green one, but not so much from the word *dog* to the word *cat*, so backpropagation seems to bemore complex.
    - Images have spatial dimension, text has time/sequential dimension, even with long-term depencencies.
    - Text data has rule-based grammatical structure.
- Processing:
  - Tokenization: splitting text into individual units
    - Often lowercased.
    - Vocabulary is created; `unknown: [UNK]` token for terms not present in traning vocabulary.
    - Stemming can be performed: reduce to simplest form.
    - Punctuation can be tokenized.
    - In some cases *stop words* are removed.
    - Tokens have and interger id.
    - Special tokens: `[UNK]: 1`, `[EOS]: 0`
  - Sequence generation: a sequence length is chosen, then text items (i.e., recipes) are either truncated or padded.
    - Vocabulary size: 10k; rest `unknown: [UNK]`
    - Sequence length: 200; rest truncate or pad (latter sequence items).
    - Sequence pairs are generated
      - The sequence itself (input).
      - The same sequence shifted one token (target).
    - **Even though we set a fied sequence length, the model can work with any sequence lengths, because of how it is constructed!**
- Model
  - Embedding layer: lookup table in which a row (token id) contains an embedding vector of a predefined length (embedding dimension).
    - The vectors can be learned though backpropagation and they can be even semantic, i.e., similar words have similar vectors.
    - The embedding layer has many parameters! Example: 10k vocabulary size x 100 embedding dim: 1M weights.
    - Continuous vector representations for tokens are much better: more compressed, optimizable through backpropagation.
  - LSTM layer: it receives tensors of size `[batch_size, seq_len, embed_dim]`.
    - In each batch, all the vectors `x_i in {x_1, ..., x_n}` in the sequence are passed **one after the other in steps** to the LSTM cell.
    - In each step, hidden state vectors of size `hidden_dim` (aka. number of units) are created `h_i`; these vectors are passed along with the next sequence vector `x_(i+1)` to the LSTM again in the next step, until we reach the final vector in the sequence.
      - Similarly, in LSTMs, `C_i` in ternal beliefs are also created, of the same size as `h_i`; these are passed from a step to the next, but are not used as the cell's output.
    - The final sequence vector `x_n` produces the final hidden state `h_n`.
    - Therefore, an LSTM performs `seq_len = n` passes for each sequence.
- Training
  - The output layer is a vector of vocabulary size, which predicts the probabilities of the next word.
  - The model is trained with a callback cass `TextGenerator` which is able to sample from the output vector.
  - The text generator callback has access to the model, so we can predict a sequence of tokens `on_epoch_end`. The selection of the token occurs with a `sample_from` function, which takes a `temperature` argument. `temperature` is used as an inverse exponent applied to the probabilities, and then token sampling is done according to the modified probabilities; thus:
    - `t < 1`: larger probabilities become larger, smaller smaller; less *creative*, more *accurate*.
    - `t = 1`: probabilities of the model output.
    - `t > 1`: more homogeneous probabilities, i.e., large smaller and small larger; more *creative*, less *accurate*.
    ```python
    # Create a TextGenerator checkpoint
    class TextGenerator(callbacks.Callback):
        def __init__(self, index_to_word, top_k=10):
            self.index_to_word = index_to_word
            self.word_to_index = {
                word: index for index, word in enumerate(index_to_word)
            }  # <1>

        def sample_from(self, probs, temperature):  # <2>
            probs = probs ** (1 / temperature)
            probs = probs / np.sum(probs)
            return np.random.choice(len(probs), p=probs), probs

        def generate(self, start_prompt, max_tokens, temperature):
            start_tokens = [
                self.word_to_index.get(x, 1) for x in start_prompt.split()
            ]  # <3>
            sample_token = None
            info = []
            while len(start_tokens) < max_tokens and sample_token != 0:  # <4>
                x = np.array([start_tokens])
                y = self.model.predict(x, verbose=0)  # <5>
                sample_token, probs = self.sample_from(y[0][-1], temperature)  # <6>
                info.append({"prompt": start_prompt, "word_probs": probs})
                start_tokens.append(sample_token)  # <7>
                start_prompt = start_prompt + " " + self.index_to_word[sample_token]
            print(f"\ngenerated text:\n{start_prompt}\n")
            return info

        def on_epoch_end(self, epoch, logs=None):
            self.generate("recipe for", max_tokens=100, temperature=1.0)
    ```
- Improvements
  - Stacked LSTM cells: it is common to have 2-3 layers of LSTM cells one after the other; the next layer uses all the hidden states of the previous as inputs; i.e., the sequence is formed by the hidden states formed in the sequence steps.
  - GRU: Gated Recurrent Units; a modification/simplification of LSTM cells which seems to have improved performance in some cases.
  - Bidirectional cells: hidden states are doubled and the model learns to predict in reverse direction.

#### Image generation with PixelCNN

- Image generation by predicting the next pixel value given the precedeing pixels.
  - Pixel order: from top-left to bottom-right, first rows, then columns.
  - We need masked convolutional filters: we take a convolutional filter and add a pixel-wise multiplied mask to it.
    - In the mask, the center pixel can be 1 or 0, the posteriors 0, the previous 1.
  - Additionally residual blocks are used: layer input is added to the ouput before being passed to the rest of the network, i.e., *skip connections*.
    - Only difference needs to be learned.
    - We alleviate the vanisching gradient problem.
    - We can learn the identity mapping much easily.
- The training is similar to the Autoencoder: the the network tries to recreated the input image; difference: no previous pixels used.
  - The training is very slow, because the network doesn't know that a pixel value of 200 is close to 201.
  - Compromise: instead of values in `[0,255]`, pixel values in `[0,1,2,3]` are used.
- Prediction starts from an image of zeros.
- Keras implementation is interesting, because the layers are defined as classes: `ResidualBlock`, `MaskedConvLayer`.
- Improvements: Mixture Distributions
  - In the implemented example, 4 levels of pixel values are used due to the fact that training PixelCNN is expensive.
  - An improvement consists in predicting the mean and variance of distributions that are mixed; these distributions are put on the `[0,255]` range and we sample pixel value from them. Thus, much less parameters need to be predicted, but the pixel band-width is 256.

### Notebooks

- [`lstm.ipynb`](./notebooks/05_autoregressive/01_lstm/lstm.ipynb)
- [`pixelcnn.ipynb`](./notebooks/05_autoregressive/02_pixelcnn/pixelcnn.ipynb)
- [`pixelcnn_md.ipynb`](./notebooks/05_autoregressive/03_pixelcnn_md/pixelcnn_md.ipynb)

### List of papers

- LSTM (Hochreiter and Schmidhuber, 1997): [Long Short-term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
- PixelCNN (van den Oord et al., 2016): [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759v3)

## Chapter 6: Normalizing Flow Models

### Key points

In an nutshell, and in my own words:

Normalizing Flow Models were presented for the first time in the RealNVP paper (2017). The idea behind is very similar to the Variational Autoencoder, but we have two new important properties:

1. Instead of having 2 independent submodels (encoder and decoder), there's **only one which is invertible**.
2. The **latent space** they generate can have the shape we would like, **usually a Gaussian**. That makes the latent representations much more controllable (recall that was the advantage of the Variational Autoencoders wrt. the Autoencoders alone); thus, we can create new images/samples much better.

To achieve these new properties, the architecture is constructed as a series of variable changes (the input vector `x` would be the variable which is mapped to `z = f(x)`); these changes are performed so that we can compute the inverse variable change very easily:

- The variable changes apply `scale + translate` operations with factors/parameters that are obtained from layers/mappings with learned weights.
- One of the key aspects to obtain the inverse variable function `x = g(z)` is the determinant of the Jacobian of the variable change (`J = [dz/dx]`). For large variable dimensions (e.g., images), this is very expensive to compute. However, by alternating the parts of the variable where the change is applied in sequential layers, the Jacobian becomes lower triangular, so the determinant is the product of its diagonal!

In the book the *two moons* dataset it used: 2D points that form 2 arcs. The 2D points are converted to a Gaussian scatterplot and back.

The model is composed by **coupling layers** stacked one after the other:

- Input: variable `x`.
- Then, a series of coupling layers, each performing a variable change `z = f(x)` using the previous input.
- In each coupling layer, the dimension of the input is increased (e.g., from 2 to 256 in the example) for some hidden layers (e.g., 5) and finally reduced to the same dimension as the input.

Each coupling layer:

- Takes `x[:d]` and computes the vectors `s` and `t` with a learned mapping (linear, convolution, etc.).
- The rest `x[d:]` is scaled and shifted with `s` and `t`.
- The `z` is the concatenation of:
  - the unchanged `x[:d]`; it's like **masking the part `[:d]`**
  - the changed part `x[d:].*s + t`

The masked region `[:d]` used to obtain `s` and `t` is alternated from one coupling layer to the next: `[:d], [d:], [:d], ...`.

The final vector should have a Gaussian distribution.

The inverse network is very easy to compute: just undo the scale & translate from compound layer to compound layer.

The loss contains the determinant of the Jacobian, which is simply a product of the diagonal values.

### Notebooks

- [`realnvp.ipynb`](./notebooks/06_normflow/01_realnvp/realnvp.ipynb)

### List of papers

- RealNVP (Dihn et al., 2017): [Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803)
- GLOW (Diedrick et al., 2018): [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

## Chapter 7: Energy-Based Models

### Key points

I think the chapter is not that easy to understand, or at least, I was a bit confused.

I will try to summarize the most important points here.

In energy-based models, a network `E(x)` is trained to output:

- low scores that lead to `p(x) ~ 1` for dataset/likely observation;
- high scores that lead to `p(x) ~ 0` for unlikely observations.

The probability `p(x)` is modeled by the Boltzman distribution:

`p(x) = exp(-E(x)) / int(x, exp(-E(x)))`

The neural network is in practice an energy function `E(x)` which 

- takes as input an tensor `x`
- and outputs a scalar in `[-inf, inf]`.

In the chapter example, the MNIST dataset is used, so the image tensors are converted to an energy value in `[-inf, inf]` passing through some convolutional layers that use the swish activation function (a smooth ReLU).

Two central questions need to be addresse; with all the definitions done so far

1. How do we sample new images with low energy score? Langevin dynamics is used for that.
2. How can we deal with the intractable integral of the `p(x)`? Contrastive divergende is used for that.

**Langevin dynamics** is used to sample new images or `x` tensors. During training, instead of computing the gradient of the loss function wrt. the network parameters, now:

- we keep the network parameters fixed
- and we compute the gradient of the output wrt. the input `x`.
- Then, we update the input (initially random) in the direction of the negative gradient in several steps.
- Thus, we obtain an image which has low energy score, i.e., it is likely.

The training is done using **contrastive divergence**. This technique achieves to write the loss function as a function of the energy and without the intractable integral in the Boltzman energy distribution.

In the notebook a `generate_sample()` function is defined which is able to iteratively generate a realistic sample from a noise tensor.

### Notebook

[`ebm.ipynb`](./notebooks/07_ebm/01_ebm/ebm.ipynb)

### List of papers

- Energy Based Models (Du et al., 2019): [Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)

## Chapter 8: Diffusion Models

### Key points

Diffusion models outperform the previous GAN models for image generation.

The core idea is that we train a model which takes

- a noisy image (in the beggining it will be a pure random noise map)
- and an associated noise variance (in the beginning it will be a high variance value)

and it predicts the noise map overlaid on the image, so that we can substract it to from the noisy image get the noise-free image.

The process is performed in small, gradual steps, and following a noise rate schedule.

We can differentiate these operation phases:

- During **training**, the original image is modified: we add a noise map related to a variance value to the image and pass the image to a U-Net model which predicts the noise map; the error is backpropagated, so that the model parametrizes the noise contained in an image. This is done gradually in around `T = 1000` steps in which the noise is added following a cosine schedule. The process of gradually adding noise is called **forward diffusion**.
- During **inference**, the U-Net is used to predict the noise map, starting from a random noise map. In around `T = 20` steps, the noise map is predicted and substracted from the image, feeding the new image (with less noise) to the U-Net again to predict the next noise step. The process of gradually removing noise is called **reverse diffusion** or **denoising**.
  - It is possible to interpolate between latent Gaussian noise maps, i.e., first, we blend one noise map into another in `n` steps using a sinusoidan function; then, for blended noise map of a step we run the reverse diffusion function, i.e., the denoising. The result is that the final images interpolate accordingly (also progressively) from one to the other.

We can see that in any operation phase (training, inference) the number of forward passes is linear with the steps taken for adding noise or denoising.

#### More details

The **forward diffusion** function `q` adds the required noise between two consecutive noisy images (`x_(t-1) -> x_(t)`); it is defined as:

    x_t = q(x_t | x_(t-1)) = sqrt(1-b_t) * x_(t-1) + sqrt(b_t) * e_(t-1)
  
where

    x: image
    t: step in noise adding schedule
    b, beta: variance
    e, epsilon: Gaussian map with mean 0, standard deviation 1

However, a **reparametrization trick** allows to formulate the function such as any stage of the noisy image (`t`) can be computed from the original, noise-free image (`x_0`):

    x_t = q(x_t | x_0) = sqrt(m(a_t)) * x_0 + sqrt(1 - m(a_t)) * e_t

where

    a_t, alpla_t = 1 - b_t
    m(a_t) = prod(i=0:t; a_i)

Note that 

- the value `m(a_t)` is the **signal** ratio
- whereas the `1 - m(a_t)` is the **noise** ratio.

Additionally, `e_t` is exactly what the U-Net model is trying to output given `b_t` and the noisy image!

Diffusion schedules vary the signal and noise ratios in such a way that

- the signal ratio decreases (i.e., increase the value of `t` in `m(a_t)`) from `1-offset` to 0 following a cosine function
- and the noise ratio increases from `offset` to 1 folllowing the complementary cosine function.

The **reverse diffusion** function `p` removes noise between two consecutive noisy images (`x_(t) -> x_(t-1)`); it has this form:

    x_(t-1) = p(x_(t-1) | x_t) = f(m(a_t); x_t)

This **reverse diffusion** function is derived from the reparametrized forward diffusion and other concepts; it has a simple linear form but fractional coefficients dependent on the signal ratio `1 - m(a_t)` and the noise ratio `m(a_t)`. More importantly, **it uses the noise map `e` which is predicted by the trained U-Net, i.e., the U-Net is trained using the forward diffusion to be able to create the necessary noise map value to be substracted in the reverse diffusion.** 

The **U-Net noise model** has the following properties:

- Input: noisy image `x_t` at step `t`, as well as variance `b_t`
- Output: noise map `e_t` corresponding to the input; if we substract `e_t` to the noisy image `x_t` we should obtain the noise-free image `x_0`. However, obviously, that works better if done progressively in the reverse diffusion function.
- 


### Notebooks

Notebook: [`ddm.ipynb`](./notebooks/08_diffusion/01_ddm/ddm.ipynb).

I was able to run part of the notebook on Google Colab; I got RAM error/saturation in the first epoch (approximately in the middle of it) using an L4 GPU.

In the notebook, the Oxford 102 flowers dataset is used, which consists of 8k flower images (in color), which are resized to `64 x 64 x 3` with pixel values in the range `[0,1]`.

### List of papers and links

- Denoising Diffusion Probabilistic Models (Ho et al., 2020): [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
  - Original Diffusion paper. This paper is implemented, with some improvements from posterior works.
- Improved Diffusion (Nichol et al.): [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
  - Cosine diffusion schedule presented here.
- Transformers (Vaswani et al.): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Transformers paper.
  - Sinusoidal embeddings are presented, but for encoding position.
- NeRF (Mildenhall et al., 2020): [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
  - Sinusoidal embeddings that map scalars to n-dimensional vectors are presented.
- Implicit Diffusion (Song et al., 2020): [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
  - The deterministic denoising method is presented, with a well defined `t -> t-1` denoising formula; this formula is used.

## Chapter 9: Transformers

### Key points

### Notebooks

### List of papers and links

## Chapter 10: Advanced GANs

### Key points

### Notebooks

### List of papers and links

## Chapter 11: Music Generation

## Chapter 12: World Models

## Chapter 13: Multimodal Models

## Chapter 14: Conclusion


