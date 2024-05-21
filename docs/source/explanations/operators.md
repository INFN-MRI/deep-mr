# MR Encoding operator
The encoding operator $E$ in MRI maps the spatial distribution of NMR signals to k-space data. It generally consists of a composition of linear and non-linear operators:

$$
E(x) = S \cdot F_T \cdot C \cdot B(x)
$$

where:
- $B$ represents the non-linear Bloch response,
- $C$ represents the coil sensitivity profiles,
- $F_T$ represents the (Non-Uniform) Fourier transform,
- $S$ represent the selection operator of actual samples.

Nonlinear components, such as the Bloch response, can be expressed as \(\mathbf{B}(\mathbf{x})\). These can either be absorbed into the target image for contrast-weighted imaging:

\[
\mathbf{y} = \mathbf{C} \mathbf{F}_T \mathbf{B}(\mathbf{x})
\]

or replaced by subspace projection for low-rank inversion or low-rank tensor formulation:

\[
\mathbf{B}(\mathbf{x}) \approx \mathbf{U} \mathbf{x}
\]

where \(\mathbf{U}\) is the subspace basis. This leads to:

\[
\mathbf{y} = \mathbf{C} \mathbf{F}_T \mathbf{U} \mathbf{x}
\]