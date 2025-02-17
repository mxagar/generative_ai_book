# Generative Deep Learning: My Notes

These are my notes of the book [Generative Deep Learning, 2nd Edition, by David Foster (O'Reilly)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/).

![Book Cover](./img/book_cover.png)

This repository is a fork of the [official book repository](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition). I modified the notebook contents and moved the original `README.md` to [README_ORG.md](README_ORG.md).

Some related repositories of mine:

- My personal notes on the O'Reilly book [Natural Language Processing with Transformers, by Lewis Tunstall, Leandro von Werra and Thomas Wolf (O'Reilly)](https://github.com/mxagar/nlp_with_transformers_nbs)
- My personal notes on the [Udacity Generative AI Nanodegree](https://github.com/mxagar/generative_ai_udacity)
- My personal hands-on guide on [Hugging Face](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- My personal hands-on guide on [LangChain](https://github.com/mxagar/tool_guides/tree/master/langchain)
- My personal hands-on guide on *"classical"* approaches in [Natural Language Processing (NLP)](https://github.com/mxagar/nlp_guide) 
- Some personal notes on tools for working with [Large Language Models (LLMs)](https://github.com/mxagar/tool_guides/tree/master/llms)
- Some personal notes on the basics of Deep Learning:
  - [mxagar/computer_vision_udacity/CVND_Advanced_CV_and_DL.md](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md)
  - [mxagar/deep_learning_udacity/DLND_RNNs.md](https://github.com/mxagar/deep_learning_udacity/blob/main/04_RNN/DLND_RNNs.md)

## Table of Contents

- [Generative Deep Learning: My Notes](#generative-deep-learning-my-notes)
  - [Table of Contents](#table-of-contents)
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
      - [Data Preprocessing](#data-preprocessing)
      - [Architecture Building Blocks](#architecture-building-blocks)
      - [ChatGPT and Reinforcement Learning from Human Feedback (RLHF)](#chatgpt-and-reinforcement-learning-from-human-feedback-rlhf)
        - [How is the reward model trained to produce a score if the annotation is a ranking?](#how-is-the-reward-model-trained-to-produce-a-score-if-the-annotation-is-a-ranking)
        - [How does the PPO algorithm work? The reward model produces a reward, not an error, so how can we update the weights?](#how-does-the-ppo-algorithm-work-the-reward-model-produces-a-reward-not-an-error-so-how-can-we-update-the-weights)
    - [Notebooks](#notebooks-6)
    - [List of papers and links](#list-of-papers-and-links-1)
  - [Chapter 10: Advanced GANs](#chapter-10-advanced-gans)
  - [Chapter 11: Music Generation](#chapter-11-music-generation)
  - [Chapter 12: World Models](#chapter-12-world-models)
    - [Key points](#key-points-9)
    - [List of papers and links](#list-of-papers-and-links-2)
  - [Chapter 13: Multimodal Models](#chapter-13-multimodal-models)
    - [CLIP (OpenAI, February 2021)](#clip-openai-february-2021)
    - [Dalle 2 (OpenAI, April 2022)](#dalle-2-openai-april-2022)
    - [Other Models](#other-models)
    - [List of papers and links](#list-of-papers-and-links-3)
  - [Chapter 14: Conclusion](#chapter-14-conclusion)
  - [License](#license)

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

- a noisy image `x_t` (in the beggining it will be a pure random noise map)
- and an associated noise variance `b_t` (in the beginning it will be a high variance value)

and it predicts the noise map `e_t` overlaid on the image, so that we can substract it to from the noisy image get the noise-free image `x_0`.

The process is performed in small, gradual steps, and following a noise rate schedule.

We can differentiate these operation phases:

- During **training**, the original image is modified before passing it to the model: we add a noise map related to a variance value to the image and pass the image to a U-Net model which tries to predict the added noise map; the error is backpropagated, so that the model parametrizes the noise contained in an image. This is done gradually in around `T = 1000` steps in which the noise is added following a cosine schedule. The process of gradually adding noise is called **forward diffusion**.
- During **inference**, the U-Net is used to predict the noise map, starting from a random noise map. In around `T = 20-100` steps, the noise map is predicted and substracted from the image, feeding the new image (with less noise) to the U-Net again to predict the next noise step. The process of gradually removing noise is called **reverse diffusion** or **denoising**.
  - It is possible to interpolate between latent Gaussian noise maps, i.e., first, we blend one noise map into another in `n` steps using a sinusoidan function; then, for blended noise map of a step we run the reverse diffusion function, i.e., the denoising. The result is that the final images interpolate accordingly (also progressively) from one to the other.

We can see that in any operation phase (training, inference) the number of forward passes is linear with the steps taken for adding noise or denoising.

Some corollary notes on training and inference:

- In the forward diffusion process (traning of the U-Net), the noise map (`e_t, epsilon`) is computed from the variance scalar (`b_t, beta`) and added to the image to obtain a noisy image `x_t`. Then, the noisy image is passed to the U-Net. The U-Net tries to guess the noise map, and the prediction error is used to update the weights via backpropagation.
- In the reverse diffusion process (inference), the noise map is not computed using a formula which depends on the variance, but it is predicted by the U-Net model. Then, this noise map is substracted to the image to remove a noise step from it.

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

Diffusion schedules vary the signal and noise ratios in such a way that during training

- the signal ratio decreases (i.e., increase the value of `t` in `m(a_t)`) from `1-offset` to 0 following a cosine function
- and the noise ratio increases from `offset` to 1 folllowing the complementary cosine function.

During inference or image generation the schedule is reversed.

The **reverse diffusion** function `p` removes noise between two consecutive noisy images (`x_(t) -> x_(t-1)`); it has this form:

    x_(t-1) = p(x_(t-1) | x_t) = f(m(a_t); x_t)

This **reverse diffusion** function is derived from the reparametrized forward diffusion and other concepts; it has a simple linear form but fractional coefficients dependent on the signal ratio `1 - m(a_t)` and the noise ratio `m(a_t)`. More importantly, **it uses the noise map `e` which is predicted by the trained U-Net, i.e., the U-Net is trained using the forward diffusion to be able to create the necessary noise map value to be substracted in the reverse diffusion.** 

It's worth mentioning that both the forward and reverse diffusion processes are Gaussian, meaning that the noise added in the forward process and removed in the reverse process is Gaussian. This Gaussian structure allows the formulation of the reverse process based on Bayes' theorem.

The **U-Net noise model** has the following properties:

- Input: noisy image `x_t` at step `t`, as well as variance `b_t`.
  - The variance scalar is expanded to be a vector using **sinusoidal embedding**. Sinusoidal embedding is basically a `R -> R^n` map which for each unique scalar generates a unique and different vector. It is related to the sinusoidal embedding from the Transformers paper, but there it was used to add positional embeddings. Later, in the NeRF paper, sinusoidal embeddings were modified to map scalars to vectors, as done in the diffusion U-Net model.
  - The image and the variance vector are concatenated in the beginning of the network.
- Output: noise map `e_t` corresponding to the input; if we substract `e_t` to the noisy image `x_t` we should obtain the noise-free image `x_0`. However, obviously, that works better if done progressively in the reverse diffusion function.
- As in every U-Net, the initial tensor is progressively reduced in spatial size while its channels are increased; then, the reduced vector is expanded to have a bigger spatial size but less channels. The final tensor has the same shape as the input image. The architecture consists of these blocks:
  - `ResidualBlock`: basic block used everywhere which performs batch normalization and 2 convolutions, while adding a skip connection between input and output, as presented in the ResNet architecture. Residual blocks learn the identity map and allow for deeper network, since the vanishing gradient issue is alleviated.
  - `DownBlock`: two `ResidualBlocks` are used and an average pooling so that the image size is decreased and the channels are increased.
  - `UpBlock`: upsampling is applied to the image to increase its spatial size and two `ResidualBlocks` are applied so that the channels are decreased.
  - Skip connections: the ouput of each `ResidualBlocks` in a `DownBlock` is passed to the associated `UpBlock` with same tensor size, where the tensors are concatenated.
- Two networks are maintained: the usual one with the weights computed during gradient descend and the *Exponential Moving Average (EMA)* network, which contains the EMA of the weights. The EMA network is not that susceptible to spikes and fluctuations.

![Diffusion Models](./assets/diffusion_models.jpg)

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

Check my through explanation of how Transformers work, done in [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs).

This chapter introduces the same topic, but:

- it focuses on the Generator/Decoder part of the Transformer architecture (while the notes from the link focus on the Encoder part),
- and it uses Keras to build a model (while the notes from the link use Pytorch).

### Key points

Until 2017, LSTMs were the default architecture to deal with sequential data. Then, the Transformer architecture was presented by Vaswani et al. and everything changed.

The Transformer architecture is remarkable because:

- It is quite a simple architecture (mostly composed of linear/dense layers), which leverages the attention mechanism very succesfully.
- It can process sequences at once, in contrast to LSTMs, which need to process the data sequentially. Therefore, Transformers are easily parallelizable.
- It has shown a remarkable performance when scaled in terms of size (number of parameters) and training data.

The complete Transformer architecture has two parts:

- The Encoder, which takes token sequences and encodes them into embedding vectors of a given size. These embedding vectors are often called contextualized embeddings.
- The Decoder, which takes its own generated output tokens (starting with `<start>`) and generates the next token, which can be appended to the output sequence to be fed to the Decoder again.

![Transformer Architecture (Vaswani et al., 2017)](./assets/transformer_architecture.png)

The Transformer works in summary as follows:

- We tokenize an input sequence.
- The input sequence is transformed to embedding vectors; positional embedding vectors are added, too. These can be pre-computed sinusoidal (originally) or learned (used nowadays in practice).
- These embedding vectors are continuously transformed in `N` transformer blocks to become output or contextualized embeddings; note that the size of those embeddings is the same as in the beginning, but  their values have been changed.
- In each of the `N` blocks, several simple operations/layers are applied:
  - Attention layer
  - Concatenatation
  - Addition, with skip connections
  - Normalization
  - Linear or feed-forward mappings

The **attention** mechanism/layer consists in weighting the relevance of the embedding vectors in the sequence. To that end, the embedding vectors of the sequence are linearly mapped to Query, Key and Value vectors, which are used to compute vector similarities

We can distinguish two types of attention mechanisms in the Transformer:

- *Self-referential* attention: when all similarities are computed using the embedding vectors from the input sequence (Encoder).
- *Cross-referential* attention: when the similarities are computed using, in part, vectors (Query) which come from another source than the input sequence (Decoder).

We can use different parts of the Transformer:

- The Encoder only (e.g., BERT), to generate sequence embeddings, which can be used for downstream tasks, such as sequence classification or token classification. If the Encoder is trained separately, the sequence text is shown to the architecture with a masked token which needs to be predicted. This scheme is called **masked language modeling**.
- The Decoder only (e.g., GPT), to generate text given an an input sequence (prompt). If the Decoder is trained separately, we compared its outputs to a reference text. Additionally, in the attention layer we apply **causal masking** to hide future words/tokens, otherwise it would not learn.
- The full Encoder-Decoder architecture (e.g., T5), for text-to-text tasks, such as summarization and translation.

If the full Transformer is used, the contextualized embeddings that are the output from the Encoder are inserted into the middle of the Decoder. In that process, they first are mapped to be the Key and Value vectors; meanwhile, the Query vectors in the decoder are computed from the Decoder inputs.

This chapter implements a reduced version of the GPT (Generative Pre-Trained Transformer), which is the Decoder part.

#### Data Preprocessing

The [Kaggle Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) dataset is used: 130k wine reviews/descriptions + metadata (price, country, points, etc.).

The processing is as follows:

- Load dataset and concatenate text columns (description + country, etc.)
- Pad punctuation and whitespaces.
- Pass processed text to `TextVectorization`, which tokenizes it.
- Create splits of tokenized text pairs: input and ouput; output tokens are shifted one token.

#### Architecture Building Blocks

![Transformer Architecture Components](./assets/Transformer_Architecture_Components.png)

The Transformer is composed by `N` `TransformerBlock`s (`N = 12` for BERT). Each `TransformerBlock` contains:

- In the beginning, `TokenAndPositionEmbedding`; this layer adds positional embeddings of the tokens. Since token embeddings are summed in the attention layer to generate new embeddings, positional information is lost if no positional data is added explicitly.
- A `MultiHeadAttention` layer.
- Layer normalization (2x).
- Feed-forward (dense) layers.
- Skip connections; these enable deeper networks, while avoiding the gradient vanishing problem.

A `MultiHeadAttention` layer is where the **attention** mechanism is applied. It is multi-head because `M` attention blocks (`M = 12` in BERT) are inside, where each ouputs a fraction of the embedding vectors and then all are concatenated.

In each of the `M` (self-) attention layers, the sequence of input embeddings is mapped to three 3D tensors `Q, K, V` using three separate dense layers `W_Q, W_K, W_V`. These tensors keep two of the input sequence dimensions `(batch_size, seq_len)`, while the third hidden size dimension might vary:

- `Q`, Queries: it represents the task at hand, i.e., find the missing/next token. 
  - In the case of the Encoder, the `Q` tensor comes from the input sequence itself.
  - In the case of the Decoder, the `Q` tensor is formed by the output sequence that is being generated autoregressively.
- `K`, Keys: it represents the tokens in the input sequence.
  - The tensors `Q` and `K` are used to compute the similarities between the token representations. This is the attention.
- `V`, Values: unweighted contributions of each token in the input sentence into the ouput.
  - The output of the attention layer is basically `V` scaled by the similarity weights between `Q` and `K`.

The **layer normalization** provides stability during the training process, as batch normalization; both have similar formulas (`x` vectors centered and scaled with mean and variance, and we have 2 learnable parameters), but they are different:

- Batch normalization
  - Mean and variance are computed with the activations for the complete batch and per feature `x_i`.
  - It dependes on the batch size, so it is computationally more expensive.
  - Provides stability in training.
  - Used commonly in CNNs.
- Layer normalization
  - Mean and variance are computed for each data point (sample `x`) independently, considering all features within a single layer (e.g., for each token in a transformer).
  - It is independent from the batch size, so it is generally faster.
  - Provides stability in training.
  - Used commonly in Transformers, for sequential data.

#### ChatGPT and Reinforcement Learning from Human Feedback (RLHF)

OpenAI presented in Novemner 2022 ChatGPT, a GPT-3.5 model fine-tuned as explained in the in their InstructGPT paper (Ouyang et al., 2022), following the **Reinforcement Learning with Human Feedback (RLHF)** approach. This approach aligns the model to better follow written instructions.

![Reinforcement Learning with Human Feedback (Ouyang et al., 2022)](./assets/rlhf_paper_ouyang.png)

The RLHF approach has 3 steps:

1. Supervised fine-tuning (SFT)
  - Generative model fine-tuned with human-written conversation input + output pairs.
2. Reward Model (RM)
  - Generative model produces several answers to a set of prompts.
  - Human annotator ranks the ouputs from best to worst.
  - Reward model is trained to predict score according to human annotations.
3. Reinforcement Learning (RL) with Proximal Policy Optimization (PPO)
  - Define a RL environment, where
    - Policy = Generative model (GPT Decoder)
    - State = conversation history
    - Action = Generative ouput (produced text)
    - Reward = score from the Reward Model for the generated text ouput
  - The Proximal Policy Optimization (PPO) algorithm is used to optimize the Policy, i.e., the GPT model, using the reward from the generated ouput.

##### How is the reward model trained to produce a score if the annotation is a ranking?

The **Reward Model (RM)** in **RLHF** is trained to predict a scalar reward for a single output generated by the GPT model, but the training data consists of **rankings** of multiple outputs. The key is to **convert rankings into a training signal** that the reward model can use to learn to output a scalar value.

Pairwise Comparison:

- Instead of directly using the ranking as a single scalar, the training process converts a ranked set of outputs into multiple **pairwise comparisons** between the outputs.
  - For example, if the reward model is given three outputs for a prompt, ranked from best to worst: Output 1 > Output 2 > Output 3, the training process will create **pairs**: Output 1 vs. Output 2, Output 1 vs. Output 3, and Output 2 vs. Output 3.
  - Each pair is labeled such that the reward model knows which one is preferred. The model is trained to assign a **higher score** to the preferred output and a **lower score** to the less preferred output.

Training the Reward Model:

- The reward model is trained using a **pairwise ranking loss** (such as **binary cross-entropy** or **margin loss**). Given two outputs, the reward model learns to predict which one should have a higher score.
- However, we pass one text sequence `x` at a time, and obtain one reward `R(x)` at a time. Then, these two rewards are used to compute the loss for the pair.
- Specifically, for a given pair of outputs `(x_1, x_2)`, the reward model assigns scores `R(x_1)` and `R(x_2)`. The loss function then penalizes the reward model if the higher-ranked output does not receive a higher score. One commonly used formulation for this is the **logistic loss**:
  ```
  L = -log( exp(R(x_1)) / (exp(R(x_1)) + exp(R(x_2))) )
  ```
  This loss function encourages the reward model to assign higher values to better-ranked outputs.

Scalar Reward Output:

- Once the reward model is trained, it is able to **predict a scalar reward** for a single generated output by estimating how well it aligns with human preferences. Even though the model was trained on pairwise comparisons, it now has learned to map any single output to a **reward score** that reflects its quality according to human feedback.
- This is crucial because during reinforcement learning with PPO, each generated output needs to be evaluated independently, and the reward model provides the scalar reward needed for that.

##### How does the PPO algorithm work? The reward model produces a reward, not an error, so how can we update the weights?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm that updates the weights of the policy (in this case, the GPT model) using the scalar rewards produced by the **Reward Model (RM)**. 

The main challenge is that the reward model provides a **reward**, not a direct error or loss value. However, PPO still manages to update the policy’s weights using the reward signal.

Reward vs. Error in PPO:

- In **PPO**, the **reward** is not directly used as an error for backpropagation. Instead, the reward is used to compute a measure of how **advantageous** the model's action was, i.e., how much better or worse the model's generated output was compared to its usual behavior (the policy's baseline).
- PPO uses the **Advantage Function** `A(s_t, a_t)`, which measures how much better or worse the action `a_t` (the generated text output) is compared to the expected outcome. The advantage is computed as:
  ```
  A_t = R_t - V(s_t)
  ```
  where:
  - `R_t` is the reward produced by the **Reward Model** for the generated output.
  - `V(s_t)` is the **value function** (an estimate of how good the current state `s_t`, i.e., conversation history, is).
  - The value function is not computed by the reward model, but it is computed by a separate critic network, which is trained to predict the expected return for a given state. This network works in tandem with the reward model and the policy model (GPT) to compute the advantage function, which is used in the PPO algorithm to fine-tune the policy.
  - The critic (value function) network is trained to minimize the error between the predicted value `V(st_​)` and the actual return, which is calculated based on the rewards from the reward model. This is typically done using mean squared error (MSE) between the predicted value and the actual observed return:
  ```
  L_critic = (V(s_t) - R_t)^2
  ```
  - The advantage function gives the difference between the actual reward and the expected reward, effectively giving us an **error signal** (the **advantage**).

Policy Update in PPO:

- Once the **advantage** `A_t` is calculated, it is used to update the policy. In PPO, the update is performed by maximizing a **surrogate objective** that ensures small changes in the policy’s behavior, preventing large and destabilizing updates:
  ```
  L_PPO = f(A_t, r_t)
  ```
  where `r_t` is the **probability ratio** between the new and old policy’s actions, and the **clip function** ensures that the ratio doesn’t deviate too much (to maintain training stability).

- This optimization objective is used to adjust the GPT model's weights in a way that improves the likelihood of taking actions that lead to higher rewards, but without making drastic changes that might disrupt the model.

Where Does the Error Signal Come From?

- The **error signal** in PPO comes from the **advantage function** `A_t`, which is computed using the reward from the reward model and the value function. Although the reward is a scalar (not a direct loss), it contributes to the computation of the **advantage**, which drives the policy updates.
- **Backpropagation** occurs when this advantage (which acts like a pseudo-error) is used to update the weights of the GPT model, adjusting the policy to generate outputs that receive higher rewards from the reward model.

### Notebooks

Everything is implemented in [`gpt.ipynb`](./notebooks/09_transformer/gpt/gpt.ipynb).

A reduced version of the Transformer-Decoder is used, with `M = 4` (number of heads) and `N = 1` (number of blocks). The rest of the hyperparameters related to the architecture is the following:

```python
VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
BATCH_SIZE = 32
EPOCHS = 5
```

### List of papers and links

- Transformer (Vaswani et al., 2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- BERT (Devlin, 2018): [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- T5 (Raffel et al., 2019 - Google): [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- GPT-1 (Radford et al., 2018): [Improving Language Understanding by Generative Pre-Training](https://openai.com/index/language-unsupervised/)
- GPT-2 (Radford et al., 2019): [Language Models are Unsupervised Multitask Learners](https://openai.com/index/better-language-models/)
- GPT-3 (Brown et al., 2020): [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- InstructGPT, ChatGPT RLHF (Ouyang et al., 2022): [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## Chapter 10: Advanced GANs

This chapter does not include any notebooks, but it lists several papers and relevant repositories are referenced.

Different aspects that leverage GANs are introduced.

- [ProGAN: Progressive Growing of GANs for Improved Quality, Stability, and Variation (Karras et al., 2017)](https://arxiv.org/abs/1710.10196)
  - Training performed in stages, starting with 4 x 4 images; then resolution is increased progressively by adding layers.
  - Discriminator trained with 800k real images.
  - Different normalization strategies are used, which make the model more stable.
- [StyleGAN 1 and 2: A Style-Based Generator Architecture for Generative Adversarial Networks (Karras et al., 2018)](https://arxiv.org/abs/1812.04948)
  - It builds on the the ideas of the ProGAN.
  - Style transfer can be achieved.
  - StyleGAN 2 decreased artifacts.
- [Self-Attention GAN, SAGAN: Self-Attention Generative Adversarial Networks (Zhang et al., 2018)](https://arxiv.org/abs/1805.08318)
  - A method for teaching attention to ANNs that goes beyond local (CNN kernel) scope.
  - Convolutional layers learn locally and scale feature sizes; thanks to attention, it's possible to teach ANNs where to focus, given a pixel in the image.
    - The focus region might be far away in the image.
- [BigGAN: Large Scale GAN Training for High Fidelity Natural Image Synthesis (Brock et al., 2018)](https://arxiv.org/abs/1809.11096)
  - Builds on the ideas of the SAGAN.
  - The latent distribution is a truncated Gaussian; the more extreme the truncation, the believability increases, but the variaty decreases.
  - The network is bigger than SAGAN.
- [VQ-GAN: Taming Transformers for High-Resolution Image Synthesis (Esser et al., 2020)](https://arxiv.org/abs/2012.09841)
  - VQ-GAN: Vector Quantized GAN; it builds the same idea published by the VQ-VAEs, but for GANs.
  - A *codebook* of discrete latent vectors is learned, i.e., a set of predefined vectors.
  - Then, each latent vector is mapped to the closest vector in this codebook.
  - Images are reconstructed from sequences of these quantized vectors, the number of different sequences is so large that the model can generate a virtually infinite number of images.
  - The vectors in the codebook are like learned concepts.
  - Despite operating in a discrete latent space, interpolation between discrete latent codes is possible. The model can smoothly transition between different discrete codes, allowing for continuous variation in the generated images.
- [ViT VQ-GAN: Vector-quantized Image Modeling with Improved VQGAN (Yu et al., 2021)](https://arxiv.org/abs/2110.04627)
  - Same idea a VQ-GAN, but Vision Transformers (ViT) are used instead of CNNs.

## Chapter 11: Music Generation

I skipped this chapter, since I am currently not that interested in the topic.

Broadly, two approached for music generation are presented (each with associated notebooks):

- [OpenAI's MuseNet](https://openai.com/index/musenet/): a Transformer architecture which learns to predict the next note (i.e., equivalent to a token) in a sequence based on the previous ones.
- [MuseGAN](https://arxiv.org/abs/1709.06298): a GAN which generates entire musical tracks at once by treating music as images (pitch vs. time).

One important aspect is that music has special characteristcis:

- We need to create pitch and rhythm.
- It can be monophonic (e.g., MIDI files) vs. polyphonnic (i.e., several streams of notes played simultaneously by different instruments).
- Polyphonic music creates harmonies, which can be dissonant (clashing) or consonant (harmonious).

Notebooks:

- [`transformer.ipynb`](./notebooks/11_music/01_transformer/transformer.ipynb)
- [`musegan.ipynb`](./notebooks/11_music/02_musegan/musegan.ipynb)

## Chapter 12: World Models

### Key points

This chapter is about the World Models paper (Ha and Schmidhuber, 2018): [World Models](https://arxiv.org/abs/1803.10122). No notebook is provided, but there is an official repository we can check: [WorldModels Github](https://github.com/zacwellmer/WorldModels).

The paper showed that it is possible to train a model to **learn how to perform a task through experimentation in its own particular dream environment!** The dream environment is basically a generative environment.

Reinforcement Learning is used, with its usual components:

- Environment: world, physics/game engine; it behaves according to some rules and dictates the reward.
- Agent: takes actions in the environment and receives rewards depending on the objective of the game. In the beginning, actions are random and yield bad rewards. The model in the agent is improved to generate better actions which optimize the accumulated reward.
- Game state, and/or observation, `s`.
- Action, `a`.
- Reward, `r`.
- Episode: one run of the agent in the environment, from begining to game end.

[Gymnasium](https://gymnasium.farama.org/index.html) is used with a **car racing** scenario, which is a maintained fork from the OpenAI's Gym, not developed anymore.

- Game state, `s`: 64x64 RGB image of the track + car; the track is made by pavement `N` tiles, which are not visible, but pave the way.
- Action, `a`: 3 floats: direction `[-1,1]`, acceleration `[0,1]`, braking `[0,1]`.
- Reward, `r`:
  - `-0.1` for each time step taken
  - `1000/N` if new track tile is stepped, where `N` is the total number of tiles of the track.
- Episode: game finishes if
  - 3000 steps
  - car drives off the limit
  - car reaches the end

The implicit goal is for the car to drive the track to the end.

The architecture is composed by the following elements:

- The observations are processed by a **VAE** to generate latent `z` vectors of 32D; in reality 2 latent vectors are generated, i.e., `mu` and `logvar`, which are then sampled as if they describe a Gaussian to obtain `z`.
  - The VAE is trained in an unsupervised manner, simply inputing step images and using the reconstruction loss.
- The latent `z` vector is the input to a **RNN** which, given `z`, the action and the previous reward, predicts the next state and reward using a mixture density network (MDN) as head.
  ```
  r_(t-1), z_t (mu, logvar), a_t -> [MDN-RNN] -> z_(t+1) (mu, logvar, logpi), r_t
  ```
  - A MDN is used because we want to predict the distribution of next latent state.
    - With an MDN we can predict probability distributions, rather than concrete values.
    - Environment dynamics is often multi-modal (i.e., there exist several possible future states), and a MDN models that.
    - Using MDNs helps model the stochasticity of the environment.
  - The hidden state `h` dimension of the LSTM is 256.
  - This hidden state `h` models the environment state and dynamics.
  - The MDN outputs: 
    - 5 basis distributions, each with three vectors of size 32: `mu`, `logvar` and `logpi` (probability of the distrinbution of being chosen)
    - and one value for the reward.
  - Thus, the output form the hidden vector of `256` units is mapped to size is `32 x 3 + 1 = 481` using a dense layer.
  - Then, one `z` value can be sampled from the `32 x 3` distributions.
  - The MDN-RNN is trained in a supervised manner, simply using experimented game steps.
- The **Controller** is the agent that decides the action, aka. the **Policy Network**; it is a simple dense layer which maps the current state `z` and the hidden state `h` to the action `a` that needs to be taken:
  ```
  z_t, h -> [Controller] -> a
  ```
  - The input size is `32 + 256 = 288` and the output `3`.
    - The dense layer has `288 x 3 + 3 (bias) = 867` parameters.
  - The Controller is trained with the **Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES)**, i.e., not in a un/supervised way.
    - RL algorithms are often trained with evolutionary algorithms, because there is no clear loss function which can be used to propagate an error.
 
Training protocol:

- Collect random episode data.
- Train the VAE.
- Train the MDN-RNN.
- Train the Controller.

To train the Controller, the **Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES)** is used:

- The Controller is a dense/linear layer with 867 parameters which maps `z_t, h` (288D) to `a` (3D). We want to optimize those parameters `theta` for maximum reward, but there is no clear loss/error.
- We create a population of several `theta` parameter sets; in the beginning, they are randomly initialized. Each `theta` set is basically an instance of an agent with a specific controller.
- We let all the agents run the simulation and collet the results (rewards); this can be done in parallel!
- We take the 25% of the best performing agents/controllers.
  - With them, the mean and variance of the subset of the population is computed.
- With the subset mean and variance for the parameters, new parameter sets are breeded.
  - Noise is added, to allow also exploration.
  - This is like simulating Gradient Descend, but without computing the gradient wrt. any error!
- The process is repeated until convergence or maximum number of steps.

![Training the Controller in the Gym Environment (by Foster)](./assets/world_models_1.jpg)

Once we have a basic training (VAE and MDN-RNN), the nice thing of the setup is that we can ignore the game engine (the environment) and substitute it with the MDN-RNN model. Then, the Controller can be trained with the MDN-RNN alone, i.e., this is like a **dreaming environment which learns by generating fake but effective/coherent states**! The results were remarkable!

![Training the Controller in the MDN-RNN Dream Environment (by Foster)](./assets/world_models_2.jpg)


### List of papers and links

- World Models paper (Ha and Schmidhuber, 2018): [World Models](https://arxiv.org/abs/1803.10122)

## Chapter 13: Multimodal Models

In this chapter, no notebooks are provided, but a high level explanation of papers/works in which several modalities (images, text, etc.) are combined in the context of generative AI.

### CLIP (OpenAI, February 2021)

CLIP (Radford et al., 2021) is a model which learns the representation of images and texts (captions) in the same latent space, presented in the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020):

- CLIP is not a generative model, but it is used in Dalle 2, which is a text-to-image generative model.
- CLIP was trained with 400 million image-caption pairs scraped from the Internet; ImageNet has 14 million hand-annotated images.
- CLIP is composed by two encoders:
  - Text encode: Transformer-Encoder which encodes text captions into a sequence of embedding vectors.
  - Image encoder: Vision Transformer which encodes images into embedding vectors. In a Vision Transformer, an image is broken down to image patches, equivalent to tokens; each patch is then encoded to a embedding using CNNs, and then the typical Transformer-Encoder architecture follows.
- For each output sequence, the first token embedding is taken (`[CLS]` or classification token), because it is expected for it to be the summary of the image/text.
- During training, the ouput embedding pairs are compared with cosine similarity:
  - Real pairs should yield a large dot product.
  - Fake pairs should yield a small dot product.
  - Contrastive learning is applied, using the similarity values.
- As a result, the text and image embeddings belong to the same latent space.

![CLIP (Foster)](./assets/clip.jpg)

The remarkable thing of CLIP is that it can perform SoTA zero-shot image classification! Other models trained on specific datasets often fail! Procedure:

- We take out dataset.
- If we have `L` classification labels, we create a set of `L` text embedding vectors for the texts `this is a photo o a {label}`, passing them to the text encoder. This will be a fixed set of *text label embeddings*.
- Then, we take an image, obtain its image embedding, and compute the cosine similarity against all the *text label embeddings*.
- The largest dot product predicts the class!

The model is open source: 

- [openai/CLIP](https://github.com/openai/CLIP)
- [huggingface.co/openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

### Dalle 2 (OpenAI, April 2022)

Dalle 2 is a text-to-image paper which consists of these components:

- **CLIP** to convert texts (prompt image descriptions) into text emebddings. CLIP weights are frozen.
- A **prior network** which converts the text embedding into an image embedding. Even though both image and text embeddings should be aligned, the CLIP embeddings cannot be used to generate images out-of-the-box:
  - CLIP embeddings are discriminative, not generative: an image contains much more information than the description of its caption (style, texture, background, etc.). That is discarded by a discriminative task (alignement of embeddings).
- A **Decoder** model based on the Diffusion model GLIDE which converts the image embedding into the image.

Two types of **prior networks** were tried:

- **Autoregressive prior**: Encoder-Decoder Transformer which translates a CLIP text embedding into a CLIP image embedding.
  - The Encoder transforms the text embedding first; then, the transformed representation is passed to the Decoder (`K, V`).
  - The Decoder produces autoregressively one image embedding element value at a time: the previous vector values are used as input, and the Encoder representation is fed into the cross-attention layer.
  - As a result, a CLIP image embedding is generated.
  - This approach is worse than the diffusion prior, among others, because it works autoregressively, i.e., it's slower.
- **Diffusion prior**: A Decoder-Transformer is used in a diffusion process (instead of a U-Net) to obtain the CLIP image embedding from the CLIP text embedding.
  - This approach works better than the previous one.
  - During training, forward diffusion is applied to the CLIP image embedding: we add noise to the embedding in `T` steps.
    - For each step, a Transformer-Decoder is used to predict the added noise vector.
    - The Decoder is conditioned by the text embedding every time, i.e., we pass as external contextualized embeddings `K` and `V`, which are the result of mapping the text embedding with `W_K` and `W_V`.
  - During inference, we start with a random noise vector (to become he CLIP image embedding) and the CLIP text embedding.
    - We run the reverse diffusion process, i.e., we predict the step noise vector with the trained Transformer-Decoder step-by-step and progressively "clean" the CLIP image embedding.

The final model, the **Decoder**, is a U-Net model which creates the CLIP image embedding. The architecture is based on GLIDE (Nichol et al., 2021), a previous model from OpenAI which was able to create images from raw prompts, i.e., without CLIP embeddings.

In Dalle 2, the Decoder uses 

- (the random noise vector)
- the U-Net denoiser from GLIDE
- the Transformer text encoder
- and the predicted CLIP image embedding as conditioning.

The output of the Decoder is a `64 x 64` image; then, this image is passed through 2 **Upsampler** models which run a diffusion process to create first `256 x 256` images, and finally `1024 x 1024` images.

Notes:

- The fact that we use a random noise vector in the beggining of the Decoder, makes possible to generate variations of the same prompt!
- It is possible to work without the prior network, but the best results are generated with it!
- The image qualities are remarkable, but 
  - *attribute binding* fails, i.e., spatial and relational properties are not properly depicted in the image
  - and text cannot be reproduced.

Papers:

- Dalle 2 (Ramesh et al., April 2022): [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://cdn.openai.com/papers/dall-e-2.pdf)
- GLIDE (Nichol et al., 2021): [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)

### Other Models

- **IMAGEN** (Saharia et al., May 2022): [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)
  - Very similar to Dalle 2, but from Google.
  - No CLIP used; instead, a Transformer Encoder-Decoder (T5) is used to generated text embeddings, and the model is trained only on text.
  - Decoding diffusion is similar, but conditining is done only with text embeddings.

- **Stable** Diffusion (Rombach et al., August 2022): [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
  - From Munich! LMU and Runway
  - **In contrast to Dalle 2 and Imagen, the code and weights of Stable Diffusion are open source!**
  - It is very similar to Dalle 2; the main difference is that they added an autoendocer to the Diffision process:
    - We have CLIP (although it was later replaced by OpenCLIP, trained from scratch in Stable Diffusion 2) to produce CLIP image and text embeddings.
    - We have the Diffusion prior to generate CLIP image embeddings from CLIP text embeddings.
    - The Diffusion decoder works on a latent space (2D), i.e., we have a VAE which has learned to compress and reconstruct images; thus, the Diffusion decoder denoises a random latent vector conditioned by the CLIP text and image embeddings. When the latent representation is denoised, it is passed to the VAE-Decoder to reconstruct the image.
    - Therefore, the U-Net of the Diffusion model is much lighter!
  - Example repository: [Keras: A walk through latent space with Stable Diffusion](https://keras.io/examples/generative/random_walks_with_stable_diffusion/)

- **Flamingo** (Alayrac et al., April 2022): [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
  - Family of Vision-Language Models (VLM) that act as a bridge between pre-trained vision-only and language-only models.
  - Flamingo blends image and text information in a multi-modal style; teh user is able to ask questions about an image and the model answers, as if it were understanding and *reasoning*.

### List of papers and links

- CLIP (Radford et al., 2021): [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Dalle 2 (Ramesh et al., 2022): [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://cdn.openai.com/papers/dall-e-2.pdf)
- GLIDE (Nichol et al., 2021): [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)
- IMAGEN (Saharia et al., 2022): [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)
- Stable Diffusion (Rombach et al., 2021): [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Flamingo (Alayrac et al., 2022): [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- Chinchilla (Hoffman et al., 2022): [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- ControlNet (Zhang et al., 2023): [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- PaLM (Chowdhery et al., 2023): [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

## Chapter 14: Conclusion

Several diagrams of models against date, size, etc. are shown.

## License

Please, check the original [`README_ORG.md`](./README_ORG.md).
