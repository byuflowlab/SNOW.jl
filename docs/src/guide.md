# Guide

## Derivatives (Dense Jacobians)

SNOW wraps multiple packages to make it easier to efficiently compute derivatives. For dense Jacobians switching between derivative methods is straightforward and just requires selecting different options (and ensuring that your code is compatible with the method of choice).

Caches are created pre-optimization so that zero (or near zero) allocations occur during optimization.  If you wish to fully take advantage of this speed, your own code should also not allocate.  We also reuse outputs where possible to avoid additional unnecessary function calls.


```@example dense
using SNOW

# forward finite difference
options = Options(derivatives=FD("forward"))

# central finite difference
options = Options(derivatives=FD("central"))

# complex step
options = Options(derivatives=FD("complex"))

# forward-mode algorithmic differentiation
options = Options(derivatives=ForwardAD())

# reverse-mode algorithmic differentiation
options = Options(derivatives=ReverseAD())

# Zygote (reverse-mode source-to-source AD)
options = Options(derivatives=RevZyg())

nothing #hide
```

Note that dense Jacobians are the default so the above are equivalent to explicitly specifying dense:

```@example dense
options = Options(derivatives=ForwardAD(), sparsity=DensePattern())
nothing #hide
```

You can also specify your own derivatives, by setting this option:
```@example dense
options = Options(derivatives=UserDeriv())
nothing #hide
```
The function signature you provide should also be modified as follows:
```julia
f = func!(g, df, dg, x)
```
with the two new inputs: df and dg that should be modified in place.  `df` is a vector containing the objective gradient ``d f / {dx}_j`` and  `dg` is a matrix containing the constraint jacobian ``{dg}_i / {dx}_j``. Below is a simple example:

```@example userdense
using SNOW

function deriv(g, df, dg, x)

    # objective
    f = x[1]^2 - 0.5*x[1] - x[2] - 2

    # constraints
    g[1] = x[1]^2 - 4*x[1] + x[2] + 1
    g[2] = 0.5*x[1]^2 + x[2]^2 - x[1] - 4

    # gradient
    df[1] = 2*x[1] - 0.5
    df[2] = -1

    # jacobian
    dg[1, 1] = 2*x[1] - 4
    dg[1, 2] = 1
    dg[2, 1] = x[1] - 1
    dg[2, 2] = 2*x[2]

end

x0 = [1.0; 1.0]  # starting point
lx = [-5.0, -5]  # lower bounds on x
ux = [5.0, 5]  # upper bounds on x
ng = 2  # number of constraints
lg = -Inf*one(ng)  # lower bounds on g
ug = zeros(ng)  # upper bounds on g
options = Options(derivatives=UserDeriv())

xopt, fopt, info = minimize(deriv, x0, ng, lx, ux, lg, ug, options)
```


## Derivatives (Sparse Jacobians)

Several methods exist to help detect sparsity patterns.  Given a function and provided bounds on `x`, the following  method generates three random points within the bounds and computes the Jacobian using either ForwardDiff or finite differencing. Elements of the Jacobian that are zero at all three spots are assumed to always be zero.  The resulting sparsity pattern is returned.  

```@example sp
using SNOW

# an example with a sparse Jacobian
function example(g, x)
    f = 1.0

    g[1] = x[1]^2 + x[4]
    g[2] = 3*x[2]
    g[3] = x[2]*x[4]
    g[4] = x[1]^3 + x[3]^3 + x[5]
end

ng = 4  # number of constraints
lx = -5*ones(5)  # lower bounds for x
ux = 5*ones(5)  # upper bounds for x
sp  = SparsePattern(ForwardAD(), example, ng, lx, ux)
```

or with central differencing

```@example sp
sp = SparsePattern(FD("central"), example, ng, lx, ux)
```

Alternative approach include specifying the three points directly:
```@example sp
x1 = rand(5)
x2 = rand(5)
x3 = rand(5)
sp = SparsePattern(ForwardAD(), example, ng, x1, x2, x3)
```

You can also provide a Jacobian directly either in dense or sparse format.
```julia
sp = SparsePattern(A)  # where A is a Matrix or a SparseMatrixCSC
```

With a provided sparsity pattern, the package can use graph coloring to reduce the number of function calls if applicable, and pass the sparse structure to the optimizer.  The format is similar to the dense case, except you provide two differentiation methods: one for the gradient, and one for the Jacobian (though you could use the same method for both).  Even when the constraint Jacobian is sparse, the function gradient is often dense.  The function gradient in that case is well suited for a reverse mode (if forward mode was used you'd require ``n_x`` forward negating the benefits of Jacobian sparsity).

```@example sp
options = Options(sparsity=sp, derivatives=[ReverseAD(), ForwardAD()])  # reverse for gradient, sparse forward for Jacobian
nothing #hide
```

Currently supported options are ReverseAD, or RevZyg for the gradient, and ForwardAD or FD for the Jacobian.

You can also provide your own derivatives.  For sparse Jacobians there are a wide variety of possible use cases (structure you can take advantage of, mixed-mode AD, using a combination of analytic and AD, etc.), and so for best performance you may want to provide your own.
```@example sp
options = Options(sparsity=sp, derivatives=UserDeriv())
nothing #hide
```
The function signature is modified like the dense case:
```julia
f = func!(g, df, dg, x)
```
except dg is a vector (not a matrix) containing the constraint jacobian in the order specified by sp.rows, sp.cols. Like the dense case, the vectors df and dg should be modified in place. 

## Algorithms

Currently, you can choose between IPOPT and SNOPT, although the latter required a paid license.  Both methods produce a *.out file defaulting to ipopt.out for the former and snopt-print.out and snopt.summary.out for the latter. The names of the files can be changed through the algorithm-specific options.

IPOPT takes an optional argument, a dictionary, where ipopt-specific options can be provided.  See a list of options [here](https://coin-or.github.io/Ipopt/OPTIONS.html).

```@example ipoptions
using SNOW

function simple!(g, x)
    # objective
    f = 4*x[1]^2 - x[1] - x[2] - 2.5
    
    # constraints
    g[1] = -x[2]^2 + 1.5*x[1]^2 - 2*x[1] + 1
    g[2] = x[2]^2 + 2*x[1]^2 - 2*x[1] - 4.25
    g[3] = x[1]^2 + x[1]*x[2] - 10.0

    return f
end
x0 = [1.0; 2.0]  # starting point
lx = [-5.0, -5]  # lower bounds on x
ux = [5.0, 5]  # upper bounds on x
ng = 3  # number of constraints
lg = -Inf*one(ng)  # lower bounds on g
ug = zeros(ng)  # upper bounds on g

# ----- set some options ------
ip_options = Dict(
    "max_iter" => 3,
    "tol" => 1e-6
)
solver = IPOPT(ip_options)
options = Options(;solver)  

xopt, fopt, info = minimize(simple!, x0, ng, lx, ux, lg, ug, options)
```

SNOPT has three optional argument: a dictionary of snopt-specific options (see Snopt documentation), a Snopt.Names object to define custome names in the output file (see <https://github.com/byuflowlab/Snopt.jl>), and a warmstart object (explained below).

Snopt also returns a fourth output, which is a struct Snopt.Out containing information like the number of iterations, function calls, solve time, constraint values, and a warm start object.  That warm start object can be put back in as an input for a later run (it contains final values for x, f, Lagrange multipliers, etc.)

The below example shows setting options, extracting some outputs, and using a warm start.

```@example snoptions
using SNOW

function fun(g, x)

    f = x[1]^2 - x[2]

    g[1] = x[2] - 2*x[1]
    g[2] = -x[2]
end

x0 = [10.0; 10.0]
lx = [0.0; 0.0]
ux = [20.0; 50.0]
ng = 2
lg = -Inf*ones(ng)
ug = zeros(ng)

# artificially limiting the major iterations so we can restart
snopt_opt = Dict(
    "Major iterations limit" => 2
)

solver = SNOPT(options=snopt_opt)
options = Options(;solver) 

xopt, fopt, info, out = minimize(fun, x0, ng, lx, ux, lg, ug, options)
println("major iter = ", out.major_iter)
println("iterations = ", out.iterations)
println("solve time = ", out.run_time)

# warm start from where we stopped
solver = SNOPT(warmstart=out.warm)
options = Options(;solver) 
xopt, fopt, info, out = minimize(fun, x0, ng, lx, ux, lg, ug, options)
println("major iter = ", out.major_iter)
println("iterations = ", out.iterations)
```



