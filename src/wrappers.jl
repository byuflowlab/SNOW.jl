# ------- Optimization Algorithms --------

abstract type AbstractSolver end

"""
    SNOPT(;options=Dict(), names=Snopt.Names(), warmstart=nothing)

Use Snopt as the optimizer

# Arguments
- `options::Dict`: options for Snopt.  see Snopt docs.
- `names::Snopt.Names`: custom names for function and variables.
- `warmstart::Snopt.Start`: custom names for function and variables.
"""
struct SNOPT{T1,T2,T3} <: AbstractSolver 
    options::T1
    names::T2
    warmstart::T3
end

SNOPT(;options=Dict(), names=Snopt.Names(), warmstart=nothing) = SNOPT(options, names, warmstart)

"""
    IPOPT(options=Dict())

Use Ipopt as the optimizer

# Arguments
- `options::Dict`: options for Ipopt.  see Ipopt docs.
"""
struct IPOPT{TD} <: AbstractSolver 
    options::TD
end

IPOPT() = IPOPT(Dict())


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
Options(; sparsity=DensePattern(), derivatives=FD("forward"), solver=SNOPT()
    ) = Options(sparsity, derivatives, solver)


function resizebounds(lx, nx)
    if length(lx) == 1 && nx > 1
        lx = lx*ones(nx)
    end
    return lx
end


"""
    minimize(func!, x0, ng; lx=-Inf, ux=Inf, lg=-Inf, ug=0.0, options=Options())

Main function.  Minimize.

min     f
s.t.    lx < x < ux
        lg < g < ug

f = func!(g, x)
unless user derivatives then: f = func!(g, df, dg, x)

equality constraint: lg = ug
"""
function minimize(func!, x0, ng; lx=-Inf, ux=Inf, lg=-Inf, ug=0.0, options=Options())

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

    return optimize(options.solver, cache, x0, lx, ux, lg, ug, rows, cols)
    # returns xopt, fopt, info, out
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


# wrapper for IPOPT
function optimize(solver::IPOPT, cache, x0, lx, ux, lg, ug, rows, cols)
    
    # initialize
    nx = length(x0)
    ng = length(lg)
    options = solver.options
    
    # initialize data for caching results since ipopt separates functions out
    xlast = 2*x0
    f = 0.0
    g = zeros(ng)
    df = zeros(nx)
    dg = zeros(length(rows))

    function ipcon(x, g)  # ipopt calls this function first

        f = evaluate!(g, df, dg, x, cache)
        xlast = x
        
        return nothing
    end

    function ipobj(x)
        if !isequal(x, xlast)
            f = func!(g, x)
        end
        return f
    end

    function ipgrad_obj(x, grad)
        if !isequal(x, xlast)
            f = evaluate!(g, df, dg, x, cache)
            xlast = x
        end
        grad[:] = df
        return nothing
    end

    function ipjac_con(x, mode, r, c, values)
        if mode == :Structure
            r[:] = rows
            c[:] = cols
        else
            if !isequal(x, xlast)
                f = evaluate!(g, df, dg, x, cache)
                xlast = x
            end
            values[:] = dg
        end
        return nothing
    end

    nzJ = length(rows)
    nH = 1  # irrelevant for quasi-newton
    prob = Ipopt.createProblem(nx, lx, ux, ng, lg, ug, nzJ, nH, 
        ipobj, ipcon, ipgrad_obj, ipjac_con)

    # set options
    Ipopt.addOption(prob, "hessian_approximation", "limited-memory")
    for (key, value) in options
        Ipopt.addOption(prob, key, value)
    end

    # open output file
    filename = "ipopt.out"
    print_level = 5
    if haskey(options, "output_file")
        filename = options["output_file"]
    end
    if haskey(options, "print_level")
        print_level = options["print_level"]
    end
    Ipopt.openOutputFile(prob, filename, print_level)

    # solve problem
    prob.x = x0
    status = solveProblem(prob)

    return prob.x, prob.obj_val, Ipopt.ApplicationReturnStatus[status], nothing
end


# wrapper for SNOPT
function optimize(snopt::SNOPT, cache, x0, lx, ux, lg, ug, rows, cols)

    function fsnopt!(g, df, dg, x, deriv)
        f = evaluate!(g, df, dg, x, cache)
        fail = false  # TODO
        # TODO: use deriv

        return f, fail
    end

    if !isnothing(snopt.warmstart)
        x0 = snopt.warmstart
    end

    return snopta(fsnopt!, x0, lx, ux, lg, ug, rows, cols, 
        snopt.options, names=snopt.names)
end