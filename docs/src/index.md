```@meta
CurrentModule = SNOW
```

# SNOW

## QuickStart

Snopt solves optimization problems of the following form:
```math
\begin{aligned}
\text{minimize} &\quad  f(x)  \\
\text{subject to} &\quad  l_g \le g(x) \le u_g  \\
    &\quad l_x \le x \le u_x
\end{aligned}
```
Equality constraints can be specified by setting ``{l_g}_i = {u_g}_i``.  The expected function signature is:
```julia
f = func!(g, x)
```
where `g` is modified in-place.

We start by loading the package.

```@example opt1
using SNOW
```

Next, we define the function we wish to optimize.  

```@example opt1
function simple!(g, x)
    # objective
    f = 4*x[1]^2 - x[1] - x[2] - 2.5
    
    # constraints
    g[1] = -x[2]^2 + 1.5*x[1]^2 - 2*x[1] + 1
    g[2] = x[2]^2 + 2*x[1]^2 - 2*x[1] - 4.25

    return f
end
```

We now define the starting point, bounds, and options.  The default for nonlinear constraints is
```math
g(x) \le 0
```
which suits us for this problem, so we leave that as is.
```@example opt1
x0 = [1.0; 2.0]
nx = 2
lx = -5*ones(nx)
ux = 5*ones(nx)
ng = 2

options = Options(solver=IPOPT())
xopt, fopt, info = minimize(simple!, x0, ng, lx=lx, ux=ux, options=options)

println("xstar = ", xopt)
println("fstar = ", fopt)
println("info = ", info)
```


```@index
```

```@autodocs
Modules = [SNOW]
```
