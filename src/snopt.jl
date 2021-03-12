using .Snopt

export SNOPT

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