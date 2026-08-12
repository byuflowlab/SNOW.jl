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

addOption(prob, k, v::String) = Ipopt.AddIpoptStrOption(prob, k, v)
addOption(prob, k, v::Int) = Ipopt.AddIpoptIntOption(prob, k, v)
addOption(prob, k, v::Float64) = Ipopt.AddIpoptNumOption(prob, k, v)

ApplicationReturnStatus = Dict(
    0=>:Solve_Succeeded,
    1=>:Solved_To_Acceptable_Level,
    2=>:Infeasible_Problem_Detected,
    3=>:Search_Direction_Becomes_Too_Small,
    4=>:Diverging_Iterates,
    5=>:User_Requested_Stop,
    6=>:Feasible_Point_Found,
    -1=>:Maximum_Iterations_Exceeded,
    -2=>:Restoration_Failed,
    -3=>:Error_In_Step_Computation,
    -4=>:Maximum_CpuTime_Exceeded,
    -5=>:Maximum_WallTime_Exceeded,
    -10=>:Not_Enough_Degrees_Of_Freedom,
    -11=>:Invalid_Problem_Definition,
    -12=>:Invalid_Option,
    -13=>:Invalid_Number_Detected,
    -100=>:Unrecoverable_Exception,
    -101=>:NonIpopt_Exception_Thrown,
    -102=>:Insufficient_Memory,
    -199=>:Internal_Error)

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

    function ipjac_con(x, r, c, values)
        if values === nothing
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

    function eval_h end  # use hessian approximation instead

    nzJ = length(rows)
    nH = 1  # irrelevant for quasi-newton
    prob = Ipopt.CreateIpoptProblem(nx, lx, ux, ng, lg, ug, nzJ, nH, 
        ipobj, ipcon, ipgrad_obj, ipjac_con, eval_h)

    # set options
    addOption(prob, "hessian_approximation", "limited-memory")
    for (key, value) in options
        addOption(prob, key, value)
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
    Ipopt.OpenIpoptOutputFile(prob, filename, print_level)

    # solve problem
    prob.x = x0
    status = Ipopt.IpoptSolve(prob)
    
    # copy the values you need BEFORE freeing
    result_x = copy(prob.x)
    result_f = prob.obj_val
    result_status = ApplicationReturnStatus[status]

    # now it's safe to free the C-side memory
    finalize(prob)

    return prob.x, prob.obj_val, ApplicationReturnStatus[status], nothing
end


