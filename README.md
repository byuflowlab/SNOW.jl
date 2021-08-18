# Sparse Nonlinear Optimization Wrapper (SNOW)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://flow.byu.edu/SNOW.jl/index.html)
[![Build Status](https://github.com/byuflowlab/SNOW.jl/workflows/CI/badge.svg)](https://github.com/byuflowlab/SNOW.jl/actions)
<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/SNOW.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/SNOW.jl/dev)
-->

The problems we typically solve in our group are nonconvex, nonlinear, constrained, gradient-based, often computationally expensive, and sometimes have sparse Jacobians.  This package wraps derivative computation methods and optimization solvers that are well-suited to these types of problems.

Features:
- Allows easy switching between SNOPT and IPOPT from a common interface passing through all solver options, preserving output in files, and allowing warm starts (for SNOPT).
- Easy switching between various differentiation methods: ForwardDiff, ReverseDiff, Zygote, FiniteDiff (forward, central, complex step), and user-defined derivatives.
- Derivative calculations are all non-allocating during optimization.
- Outputs are also cached as applicable to avoid unnecessary function calls.
- Methods are provided to help determine sparsity patterns, sparse Jacobians can be updated efficiently with SparseDiffTools (using graph coloring), and the sparsity structure is passed to the solvers.

To Install

```julia
] add SNOW
```
