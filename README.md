# Computational Statistics and GPU Acceleration

This repository contains the full source code, data, and materials for the MSc thesis "Computational Statistics and GPU Acceleration" by Andreas S. Bampouris, submitted to the Department of Statistics at the Athens University of Economics and Business.

## Abstract

Modern statistical methods often become computationally prohibitive as data volumes and model complexity grow. This thesis examines how Graphics Processing Unit (GPU) acceleration can expand the practical scale of such methods. We organize the work around three components: (1) a theoretical analysis of computational bottlenecks in two widely-used but immensely intensive methods, Kernel Methods and Gradient Boosting, and the algorithmic redesign required for efficient GPU execution; (2) an empirical validation of the potential performance gains by benchmarking two state-of-the-art, GPU-accelerated libraries, Falkon and XGBoost, against CPU-based baselines on real-world datasets to quantify speedups and assess effects on predictive accuracy; and (3) an implementation-oriented overview of the enabling software frameworks, developing a massively parallel Markov Chain Monte Carlo (MCMC) sampler in CUDA as an illustrative case study.

Results indicate that substantial performance gains are attainable on commodity GPU hardware with no material loss in statistical accuracy when algorithms are reformulated to exploit fine-grained parallelism and memory hierarchies. More broadly, the findings underscore that scalability in statistics is as much an engineering problem as it is a methodological one: algorithm design, data layout, and hardware architecture must be considered jointly. By moving from theory, to empirical evidence, to the underlying engineering, this thesis aims to bridge the gap between advanced statistical modeling and high-performance computing, and provides the tools to not only leverage but also contribute to this expanding field.

## Repository Structure

This project is fully reproducible. The repository is organized as follows:

- **`/` (root)**: Contains the `bookdown` source files (`.Rmd`) for each chapter of the thesis, along with configuration files (`_bookdown.yml`, `_output.yml`) and the bibliography (`bibliography.bib`).
- **`docs/`**: The rendered thesis document in PDF format.
- **`data/`**: Data used for the benchmarks in Chapter 3.
- **`notebooks/`**: Jupyter notebooks used to run the benchmarks for Falkon and XGBoost.
- **`py_src/`**: Python code used by the benchmark notebooks.
- **`src/`**: CUDA C++ source code for the MCMC sampler developed in Chapter 4.
- **`img/`**: Images and figures used in the thesis.

## How to Reproduce

1.  **Thesis Document**: The thesis document can be rebuilt from the source `.Rmd` files using the `bookdown` R package.
2.  **Benchmarks**: The empirical results can be reproduced by running the Jupyter notebooks located in the `notebooks/` directory.
3.  **CUDA Sampler**: The C++ code for the MCMC sampler can be compiled using the NVIDIA CUDA Compiler (NVCC).
