# Sparse Nonlinear Optimization Wrapper (SNOW)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://byuflowlab.github.io/SNOW.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://byuflowlab.github.io/SNOW.jl/dev)
[![Build Status](https://github.com/byuflowlab/SNOW.jl/workflows/CI/badge.svg)](https://github.com/byuflowlab/SNOW.jl/actions)

The problems we are generally interested in are nonconvex, nonlinear, constrained, gradient-based, often computationally expensive, and sometimes have sparse Jacobians.  This package wraps derivative computation methods and optimization solvers that are well-suited to these types of problems.  Currently, we support the following solvers: SNOPT and IPOPT, and the following differentiation methods: ForwardDiff, ReverseDiff, FiniteDiff (forward, central, complex step), SparseDiffTools, and user-defined derivatives.  All of our functions are designed to be non-allocating, to cache work vectors, to cache outputs as applicable to avoid unnecessary function calls, and to accommodate sparse Jacobians.