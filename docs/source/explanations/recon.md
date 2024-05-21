# MRI Reconstruction as an Inverse Problem

Magnetic Resonance Imaging (MRI) is a non-invasive imaging technique that reconstructs images from k-space data, presenting it as an inverse problem. The goal is to recover an image of the object from measured frequency domain data.

The forward model of MRI can be expressed as:

$$
\mathbf{y} = \mathbf{E}(\mathbf{x})
$$

where:
- $y$ is the acquired k-space data,
- $E$ is the MR encoding operator,
- $x$ is the image to be reconstructed.

Given undersampled k-space data, the inverse problem seeks $x$ from $y$. This is typically ill-posed, requiring regularization. A generalized formulation with multiple regularization terms is:

$$
\mathbf{x} = \argmin_\mathbf{x} \lVert \mathbf{y} - \mathbf{E}(\mathbf{x}) \rVert_2^2 + \sum_{i=1}^N \lambda_i R_i(\mathbf{x})
$$

where:
- $\lVert \cdot \rVert_2$ denotes the $L_2$ norm,
- $R_i(\mathbf{x})$ are regularization terms,
- $\lambda_i$ are regularization parameters.

Advanced techniques like compressed sensing and parallel imaging allow the solution of this optimization problem, leveraging sparsity and prior knowledge to produce high-quality images from limited data. To formulate and solve this problem, we need representations for the following ingredients:

- data (k-space and image-space)
- encoding operator (linear and non-linear)
- iterative algorithms for regularized optimization

DeepMR provides implementation of each of these elements.