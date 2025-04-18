# mbipy

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Modulation based x-ray phase contrast imaging.

## Installation

Clone the repository and `cd` to the directory, then:

```shell
pip install .
```

## Usage

See the [example notebooks](examples).

## Algorithms

### Normal Integration

<details closed>
  <summary>Functional Interface</summary>

  | *function*  |   CuPy    |    JAX    |   Numba   |   NumPy   |  PyTorch  |
  | :---------: | :-------: | :-------: | :-------: | :-------: | :-------: |
  |   arnison   | &#128994; | &#128994; | &#128994; | &#128994; | &#128994; |
  | dct_poisson | &#128994; | &#128994; | &#128994; | &#128994; | &#128994; |
  | dst_poisson | &#128994; | &#128994; | &#128994; | &#128994; | &#128994; |
  |   frankot   | &#128994; | &#128994; | &#128994; | &#128994; | &#128994; |
  |   kottler   | &#128994; | &#128994; | &#128994; | &#128994; | &#128994; |
  |     li      | &#128994; | &#128308; | &#128308; | &#128994; | &#128308; |
  |  southwell  | &#10067;  | &#128308; | &#128308; | &#128994; | &#128308; |

</details>

<details closed>
  <summary>OOP Interface</summary>

  |  *class*  |   CuPy    |    JAX    |   Numba   |   NumPy   |  PyTorch  |
  | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
  |    Li     | &#128994; | &#128308; | &#128308; | &#128994; | &#128308; |
  | Southwell | &#10067;  | &#128308; | &#128308; | &#128994; | &#128308; |

</details>

### Phase Retrieval

#### Implicit

<details closed>
  <summary>Functional Interface</summary>

  | *function* |   CuPy   |    JAX    |  Numba   |   NumPy   |  PyTorch  |
  | :--------: | :------: | :-------: | :------: | :-------: | :-------: |
  |    lcs     | &#10067; | &#128994; | &#10067; | &#128994; | &#128994; |
  |   lcs_df   | &#10067; | &#128994; | &#10067; | &#128994; | &#128994; |
  |  lcs_ddf   | &#10067; | &#128994; | &#10067; | &#128994; | &#128994; |

</details>

<details closed>
  <summary>OOP Interface</summary>

  | *class* |   CuPy   |   JAX    |  Numba   |   NumPy   |  PyTorch  |
  | :-----: | :------: | :------: | :------: | :-------: | :-------: |
  |   Lcs   | &#10067; | &#10067; | &#10067; | &#128994; | &#128994; |
  |  LcsDf  | &#10067; | &#10067; | &#10067; | &#128994; | &#128994; |
  | LcsDDf  | &#10067; | &#10067; | &#10067; | &#128994; | &#128994; |

</details>

#### Explicit

<details closed>
  <summary>Functional Interface</summary>

  | function |   CuPy   |    JAX     |  Numba   |   NumPy   | PyTorch  |
  | :------: | :------: | :--------: | :------: | :-------: | :------: |
  |   umpa   | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  |   xst    | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  |   xsvt   | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  | xst_xsvt | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |

</details>

<details closed>
  <summary>OOP Interface</summary>

  | function |   CuPy   |    JAX     |  Numba   |   NumPy   | PyTorch  |
  | :------: | :------: | :--------: | :------: | :-------: | :------: |
  |   Umpa   | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  |   Xst    | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  |   Xsvt   | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |
  | XstXsvt  | &#10067; | &#10067; ¹ | &#10067; | &#128994; | &#10067; |

</details>

<!-- TODO nin17: check these are actually correct -->

¹ Has significant memory usage during compilation due to lack of strides in JAX.

² No wavelet transform available in numba - contributions welcome!

## Data

To download the data for the examples, you need to use [git-lfs](https://git-lfs.com).

## Dependencies

* [array-api-compat](https://pypi.org/project/array-api-compat/)
* [numpy](https://pypi.org/project/numpy/)

Optional:

* [scipy](https://pypi.org/project/)
* [pywavelets](https://pypi.org/project/PyWavelets/)
* [cupy](https://pypi.org/project/cupy/)
  * [pyvkfft](https://pypi.org/project/pyvkfft/)
* [numba](https://pypi.org/project/numba/)
  * [rocket-fft](https://pypi.org/project/rocket-fft/)
* [jax](https://pypi.org/project/jax/)
  * [jaxwt](https://pypi.org/project/jaxwt/)
* [torch](https://pypi.org/project/torch/)
  * [torch-dct](https://pypi.org/project/torch-dct/)
  * [ptwt](https://pypi.org/project/ptwt/)
