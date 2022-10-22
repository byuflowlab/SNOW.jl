# QuickStart

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
where ``g`` is modified in-place.  The optimizer returns ``x^*, f^*``, an information message regarding how the algorithm terminated, and potentially a struct with additional solver-specific outputs.



We start by loading the package.

```@example opt1
using SNOW
```

!!! note "Snopt" 

    If you want to use Snopt as the optimizer you need to build [Snopt.jl](https://github.com/byuflowlab/Snopt.jl) separately and load the package separate from this one ```using Snopt``` (either before or after ```using SNOW```).  It can't be loaded by default because the code is not freely available.  In this example we use Ipopt for the optimization as it is freely available.


!!! note "ReverseAD" 

Similar to "Snopt" `ReverseAD` is not available by default. You need to load the `ReverseDiff` package in order to use `ReverseAD`

Next, we define the function we wish to optimize.  

```@example opt1
function simple!(g, x)
    # objective
    f = 4*x[1]^2 - x[1] - x[2] - 2.5
    
    # constraints
    g[1] = -x[2]^2 + 1.5*x[1]^2 - 2*x[1] + 1
    g[2] = x[2]^2 + 2*x[1]^2 - 2*x[1] - 4.25
    g[3] = x[1]^2 + x[1]*x[2] - 10.0

    return f
end
nothing # hide
```

We can now optimize the function as follows:
```julia
x0 = [1.0; 2.0]  # starting point
ng = 3  # number of constraints
xopt, fopt, info = minimize(simple!, x0, ng)
```

While the defaults are suitable in this case, for the purposes of demonstration we will be more explicit about the bounds on x (defaults to -Inf to +Inf) and the bounds on the nonlinear constraints (defaults to g(x) <= 0).  We will also explicitly set the IPOPT optimizer (also a default).

```@example opt1
x0 = [1.0; 2.0]  # starting point
lx = [-5.0, -5]  # lower bounds on x
ux = [5.0, 5]  # upper bounds on x
ng = 3  # number of constraints
lg = -Inf*ones(ng)  # lower bounds on g
ug = zeros(ng)  # upper bounds on g
options = Options(solver=IPOPT())  # choosing IPOPT solver

xopt, fopt, info = minimize(simple!, x0, ng, lx, ux, lg, ug, options)

println("xstar = ", xopt)
println("fstar = ", fopt)
println("info = ", info)
```
