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


