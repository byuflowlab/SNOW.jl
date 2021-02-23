# SNOW

**Summary**: A wrapper for various sparse nonlinear optimization algorithms and derivative computation packages.

**Author**: Andrew Ning

**Features**:

- Allows easy switching between SNOPT and IPOPT from a common interface passing through all solver options, preserving output in files, and allowing warm starts (for SNOPT).
- Easy switching between various differentiation methods: ForwardDiff, ReverseDiff, Zygote, FiniteDiff (forward, central, complex step), and user-defined derivatives.
- Derivative calculations are all non-allocating during optimization.
- Outputs are also cached as applicable to avoid unnecessary function calls.
- Methods are provided to help determine sparsity patterns, sparse Jacobians can be updated efficiently with SparseDiffTools (using graph coloring), and the sparsity structure is passed to the solvers.


**Documentation**:

- Start with the [quick start](quickstart.md) to learn basic usage.
- More advanced or specific queries are addressed in the [guide](guide.md).

**Run Unit Tests**:

```julia
pkg> activate .
pkg> test
```