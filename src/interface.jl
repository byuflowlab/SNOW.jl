abstract type AbstractSolver end

function optimize(solver::AbstractSolver, cache, x0, lx, ux, lg, ug, rows, cols)
    xopt = [0.0, 0.0]
    fopt = 0.0
    info = "example"
    out = "other data"
    return xopt, fopt, info, out
end

"""
    Options(;sparsity=DensePattern(), derivatives=FD("forward"), solver=SNOPT())

Options for SNOW.  Default is dense, forward finite differencing, and SNOPT.

# Arguments
- `sparsity::AbstractSparsityPattern`: specify sparsity pattern
- `derivatives::AbstractDiffMethod`: specific differentiation methods to use
- `solver::AbstractSolver`: specificy which optimizer to use
"""
struct Options{T1,T2,T3}
    sparsity::T1  # AbstractSparsityPattern
    derivatives::T2  # AbstractDiffMethod
    solver::T3  # AbstractSolver
end

# defaults
Options(; sparsity=DensePattern(), derivatives=FD("forward"), solver=IPOPT()
    ) = Options(sparsity, derivatives, solver)


# private convenience method to size bounds of length 1 to required length
function resizebounds(lx, nx)
    if length(lx) == 1 && nx > 1
        lx = lx*ones(nx)
    end
    return lx
end

"""
    minimize(func!, x0, ng, lx=-Inf, ux=Inf, lg=-Inf, ug=0.0, options=Options())

Main function.  Minimize.

min     f
s.t.    lx < x < ux
        lg < g < ug

f = func!(g, x)
unless user derivatives then: f = func!(g, df, dg, x)

equality constraint: lg = ug
"""
function minimize(func!, x0, ng, lx=-Inf, ux=Inf, lg=-Inf, ug=0.0, options=Options())

    # initialize
    nx = length(x0)

    # parse bounds
    lx = resizebounds(lx, nx)
    ux = resizebounds(ux, nx)
    lg = resizebounds(lg, ng)
    ug = resizebounds(ug, ng)
    
    # create cache
    cache = createcache(options.sparsity, options.derivatives, func!, nx, ng)

    # determine sparsity pattern
    rows, cols = getsparsity(options.sparsity, nx, ng)

    # returns xopt, fopt, info, out
    return optimize(options.solver, cache, x0, lx, ux, lg, ug, rows, cols)
end

# -------- Alternate Interface ----------

# mutable struct Problem{TFUN, TF, TS, TI}
#     func::TFUN
#     name::TS
#     x0::Vector{TF}
#     lx::Vector{TF}
#     ux::Vector{TF}
#     xlen::Vector{TI}
#     xnames::Vector{TS}
#     lg::Vector{TF}
#     ug::Vector{TF}
#     glen::Vector{TI}
#     fnames::Vector{TS}
# end

# function createproblem(func!, name="OptProb")
#     return Problem(func!, name, Float64[], Float64[], Float64[], Int[], String[], Float64[], Float64[], Int[], String["obj"])
# end

# function add_dv!(prob, x, lx=-Inf, ux=Inf, names=[])
#     nx = length(x)
#     lx = resizebounds(lx, nx)
#     ux = resizebounds(ux, nx)
    
#     append!(prob.x0, x)
#     append!(prob.lx, lx)
#     append!(prob.ux, ux)
#     append!(prob.xlen, nx)
#     if length(names) == nx
#         append!(prob.xnames, names)
#     end
# end

# function add_con!(prob, ng, lg=-Inf, ug=0.0, names=[])
#     lg = resizebounds(lg, ng)
#     ug = resizebounds(ug, ng)
    
#     append!(prob.lg, lg)
#     append!(prob.ug, ug)
#     append!(prob.glen, ng)
#     if length(names) == ng
#         append!(prob.fnames, names)
#     end
# end

# function minimize(prob, options)

#     ng = length(prob.lg)

#     # rewrite names if applicable
#     if options.solver isa Snopt && length(prob.xnames) == nx && length(prob.fnames) == 1+ng
#         names = Snopt.Names(prob.name, prob.xnames, prob.fnames)
#         solver = Snopt(options.solver.options, names, options.solver.warmstart)
#         options = Options(options.sparsity, options.derivatives, solver)
#     end
#     return minimize(prob.func, prob.x0, ng, lx=prob.lx, ux=prob.ux, lg=prob.lg, ug=prob.ug, options=options)
# end