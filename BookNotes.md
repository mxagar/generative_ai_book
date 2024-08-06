# Generative Deep Learning: My Notes

These are my notes of the book [Generative Deep Learning, 2nd Edition, by David Foster (O'Reilly)](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/).

Table of contents:

- [Generative Deep Learning: My Notes](#generative-deep-learning-my-notes)
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


