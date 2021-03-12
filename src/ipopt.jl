# ------- IPOPT ------------

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


# ------- SNOPT --------

"""
    SNOPT(;options=Dict(), names=Snopt.Names(), warmstart=nothing)

Use Snopt as the optimizer

# Arguments
- `options::Dict`: options for Snopt.  see Snopt docs.
- `names::Snopt.Names`: custom names for function and variables.
- `warmstart::Snopt.Start`: a warmstart object (one of the outputs of Snopt.Outputs)
"""
struct SNOPT{T1,T2,T3} <: AbstractSolver 
    options::T1
    names::T2
    warmstart::T3
end

SNOPT(;options=Dict(), names=Snopt.Names(), warmstart=nothing) = SNOPT(options, names, warmstart)

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