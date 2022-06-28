


# ---------- differentiation method --------------

abstract type AbstractDiffMethod end

struct ForwardAD <: AbstractDiffMethod end
struct ReverseAD <: AbstractDiffMethod end
# struct RevZyg <: AbstractDiffMethod end  # only used for gradients (not jacobians)
struct ForwardFD <: AbstractDiffMethod end
struct CentralFD <: AbstractDiffMethod end
struct ComplexStep <: AbstractDiffMethod end
struct UserDeriv <: AbstractDiffMethod end   # user-specified derivatives

FD = Union{ForwardFD, CentralFD, ComplexStep}

"""
convert to type used in FiniteDiff package
"""
function finitediff_type(dtype)
    if isa(dtype, ForwardFD)
        fdtype = Val{:forward}
    elseif isa(dtype, CentralFD)
        fdtype = Val{:central}
    elseif isa(dtype, ComplexStep)
        fdtype = Val{:complex}
    end
    return fdtype
end


# ----- sparsity patterns -------

abstract type AbstractSparsityPattern end

struct DensePattern <: AbstractSparsityPattern end

struct SparsePattern{TI} <: AbstractSparsityPattern 
    rows::Vector{TI}
    cols::Vector{TI}
end

"""
    SparsePattern(A::SparseMatrixCSC)

construct sparse pattern from representative sparse matrix

# Arguments
- `A::SparseMatrixCSC`: sparse jacobian
"""
function SparsePattern(A::SparseMatrixCSC)
    rows, cols, _ = findnz(A)
    return SparsePattern(rows, cols)
end

"""
    SparsePattern(A::Matrix)

construct sparse pattern from representative matrix

# Arguments
- `A::Matrix`: sparse jacobian
"""
function SparsePattern(A::Matrix)
    return SparsePattern(sparse(A))
end


"""
    SparsePattern(::ForwardAD, func!, ng, x1, x2, x3)

detect sparsity pattern by computing derivatives (using forward AD)
at three different locations. Entries that are zero at all three 
spots are assumed to always be zero.

# Arguments
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `x1,x2,x3::Vector{Float}`:: three input vectors.
"""
function SparsePattern(::ForwardAD, func!, ng, x1, x2, x3)

    g = zeros(ng)
    config = ForwardDiff.JacobianConfig(func!, g, x1)
    J1 = ForwardDiff.jacobian(func!, g, x1, config)
    J2 = ForwardDiff.jacobian(func!, g, x2, config)
    J3 = ForwardDiff.jacobian(func!, g, x3, config)
    @. J1 = abs(J1) + abs(J2) + abs(J3)
    Jsp = sparse(J1)

    return SparsePattern(Jsp)
end

"""
    SparsePattern(::FD, func!, ng, x1, x2, x3)

detect sparsity pattern by computing derivatives (using finite differencing)
at three different locations. Entries that are zero at all three 
spots are assumed to always be zero.

# Arguments
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `x1,x2,x3::Vector{Float}`:: three input vectors.
"""
function SparsePattern(dtype::FD, func!, ng, x1, x2, x3)

    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(x1, zeros(ng), fdtype)

    nx = length(x1)
    J1 = zeros(ng, nx)
    J2 = zeros(ng, nx)
    J3 = zeros(ng, nx)
    FiniteDiff.finite_difference_jacobian!(J1, func!, x1, cache)
    FiniteDiff.finite_difference_jacobian!(J2, func!, x2, cache)
    FiniteDiff.finite_difference_jacobian!(J3, func!, x3, cache)

    @. J1 = abs(J1) + abs(J2) + abs(J3)
    Jsp = sparse(J1)

    return SparsePattern(Jsp)
end

"""
    SparsePattern(diffmethod, func!, ng, lx, ux)

detect sparsity pattern by computing derivatives 
at three randomly generating locations w/in bounds. 
Entries that are zero at all three spots are assumed to always be zero.

# Arguments
- `diffmethod::AbstractDiffMethod`: method to compute derivatives
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `lx::Vector{Float}`: lower bounds on x
- `ux::Vector{Float}`: upper bounds on x
"""
function SparsePattern(diffmethod, func!, ng, lx, ux)
    r = rand(3)
    x1 = @. (1-r[1])*lx + r[1]*ux
    x2 = @. (1-r[2])*lx + r[2]*ux
    x3 = @. (1-r[3])*lx + r[3]*ux
    return SparsePattern(diffmethod, func!, ng, x1, x2, x3)
end


#  used internally to get rows and cols for dense jacobian
function getsparsity(::DensePattern, nx, nf)
    len = nf*nx
    rows = [i for i = 1:nf, j = 1:nx][:]
    cols = [j for i = 1:nf, j = 1:nx][:]
    return rows, cols
end

#  used internally to get rows and cols for sparse jacobian
getsparsity(sp::SparsePattern, nx, nf) = sp.rows, sp.cols



# ---------- Dense Jacobians -------------

# internally-used cache for dense jacobians
struct DenseCache{T1,T2,T3,T4,T5}
    f!::T1  # function
    gwork::T2  # typicaly a gradient vector
    Jwork::T3   # jacobian vector/matrix
    cache::T4  # cache used by differentiation method
    dtype::T5  # diff method
end

"""
    createcache(sp::DensePattern, dtype::ForwardAD, func!, nx, ng)

Cache for dense jacobian using forward-mode AD.

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::DensePattern, dtype::ForwardAD, func!, nx, ng)

    function combine!(fg, x)
        fg[1] = func!(@view(fg[2:end]), x)
    end

    g = zeros(1 + ng)
    x = zeros(nx)
    config = ForwardDiff.JacobianConfig(combine!, g, x)
    J = DiffResults.JacobianResult(g, x)

    return DenseCache(combine!, g, J, config, dtype)
end

"""
    evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} where {T1,T2,T3,T4,T5<:ForwardAD})

evaluate function and derivatives for a dense jacobian with forward-mode AD

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} 
    where {T1,T2,T3,T4,T5<:ForwardAD})
   
    ForwardDiff.jacobian!(cache.Jwork, cache.f!, cache.gwork, x, cache.cache)
    fg = DiffResults.value(cache.Jwork)  # reference not copy
    J = DiffResults.jacobian(cache.Jwork)  # reference not copy
    f = fg[1]
    g[:] = fg[2:end]
    df[:] = J[1, :]
    dg[:] = J[2:end, :][:]
    
    return f
end


"""
    createcache(sp::DensePattern, dtype::ReverseAD, func!, nx, ng)

Cache for dense jacobian using reverse-mode AD.

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::DensePattern, dtype::ReverseAD, func!, nx, ng)

    function combine!(fg, x)
        fg[1] = func!(@view(fg[2:end]), x)
    end

    g = zeros(1 + ng)
    x = zeros(nx)

    f_tape = ReverseDiff.JacobianTape(combine!, g, x)
    cache = ReverseDiff.compile(f_tape)
    J = DiffResults.JacobianResult(g, x)

    return DenseCache(combine!, g, J, cache, dtype)
end


"""
    evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} where {T1,T2,T3,T4,T5<:ReverseAD})

evaluate function and derivatives for a dense jacobian with reverse-mode AD

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} 
    where {T1,T2,T3,T4,T5<:ReverseAD})
   
    ReverseDiff.jacobian!(cache.Jwork, cache.cache, x)
    fg = DiffResults.value(cache.Jwork)  # reference not copy
    J = DiffResults.jacobian(cache.Jwork)  # reference not copy
    f = fg[1]
    g[:] = fg[2:end]
    df[:] = J[1, :]
    dg[:] = J[2:end, :][:]
    
    return f
end


"""
    createcache(sp::DensePattern, dtype::FD, func!, nx, ng)

Cache for dense jacobian using finite differencing

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::DensePattern, dtype::FD, func!, nx, ng)

    function combine!(fg, x)
        fg[1] = func!(@view(fg[2:end]), x)
    end

    fgwork = zeros(1 + ng)
    Jwork = zeros(1 + ng, nx)
    
    x = zeros(nx)
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(x, fgwork, fdtype)

    return DenseCache(combine!, fgwork, Jwork, cache, dtype)
end


"""
    evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} where {T1,T2,T3,T4,T5<:FD})

evaluate function and derivatives for a dense jacobian with finite differencing

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} 
    where {T1,T2,T3,T4,T5<:FD})

    cache.f!(cache.gwork, x)
    f = cache.gwork[1]
    g[:] = cache.gwork[2:end]
    
    FiniteDiff.finite_difference_jacobian!(cache.Jwork, cache.f!, x, cache.cache)
    df[:] = cache.Jwork[1, :]
    dg[:] = cache.Jwork[2:end, :][:]

    return f
end

"""
    createcache(sp::DensePattern, dtype::UserDeriv, func!, nx, ng)

Cache for dense Jacobian with user-supplied derivatives

# Arguments
- `func!::function`: function of form: f = func!(g, df, dg, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::DensePattern, dtype::UserDeriv, func!, nx, ng)
    Jwork = zeros(ng, nx)
    return DenseCache(func!, 0.0, Jwork, nothing, dtype)
end

"""
    evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} where {T1,T2,T3,T4,T5<:UserDeriv})

evaluate function and derivatives for a dense jacobian with user-provided derivatives

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} 
    where {T1,T2,T3,T4,T5<:UserDeriv})
    f = cache.f!(g, df, cache.Jwork, x)
    dg[:] = cache.Jwork[:]
    
    return f
end


# ---------- gradients -------------

# internally used cache for gradients and jacobians (separately)
struct GradOrJacCache{T1,T2,T3,T4}
    f!::T1
    work::T2
    cache::T3
    dtype::T4
end

"""
    gradientcache(dtype::ReverseAD, func!, nx, ng)

Cache for gradient using ReverseDiff

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function gradientcache(dtype::ReverseAD, func!, nx, ng)

    function obj(x)
        return func!(zeros(eltype(x[1]), ng), x)
    end

    f_tape = ReverseDiff.GradientTape(obj, zeros(nx))
    cache = ReverseDiff.compile(f_tape)

    return GradOrJacCache(func!, nothing, cache, dtype)
end


"""
    gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4} where {T1,T2,T3,T4<:ReverseAD})

evaluate gradient using ReverseDiff

# Arguments
- `df::Vector{Float}`: objective gradient, modified in place
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `gradientcache`
"""
function gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4}
    where {T1,T2,T3,T4<:ReverseAD})

    ReverseDiff.gradient!(df, cache.cache, x)

    return nothing
end


# """
#     gradientcache(dtype::RevZyg, func!, nx, ng)

# Cache for gradient using Zygote (source-to-source)

# # Arguments
# - `func!::function`: function of form: f = func!(g, x)
# - `nx::Int`: number of design variables
# - `ng::Int`: number of constraints
# """
# function gradientcache(dtype::RevZyg, func!, nx, ng)

#     function obj(x)
#         return func!(zeros(ng), x)
#     end

#     return GradOrJacCache(func!, nothing, obj, dtype)
# end


# """
#     gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4} where {T1,T2,T3,T4<:RevZyg})

# evaluate gradient using Zygote

# # Arguments
# - `df::Vector{Float}`: objective gradient, modified in place
# - `x::Vector{Float}`: design variables, input
# - `cache::DenseCache`: cache generated by `gradientcache`
# """
# function gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4}
#     where {T1,T2,T3,T4<:RevZyg})

#     df[:] = Zygote.gradient(cache.cache, x)[1]

#     return nothing
# end

# function gradientcache(dtype::ForwardAD, func!, nx, ng)

#     function obj(x)
#         return func!(zeros(ng), x)
#     end

#     cache = ForwardDiff.GradientConfig(obj, zeros(nx))

#     return GradOrJacCache(obj, nothing, cache, dtype)
# end

# function gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4}
#     where {T1,T2,T3,T4<:ForwardAD})

#     ForwardDiff.gradient!(df, cache.f!, x, cache.cache)

#     return nothing
# end

# function gradientcache(dtype::FD, func!, nx, ng)

#     df = zeros(nx)
#     x = zeros(nx)
#     fdtype = finitediff_type(dtype)
#     cache = FiniteDiff.GradientCache(df, x, fdtype)

#     return GradOrJacCache(func!, nothing, cache, dtype)
# end

# function gradient!(df, x, cache::GradOrJacCache{T1,T2,T3,T4}
#     where {T1,T2,T3,T4<:FD})

#     FiniteDiff.finite_difference_gradient!(df, cache.f!, x, cache.cache)
    
#     return nothing
# end


# ------ sparse jacobians -------------

"""
    sparsejacobiancache(sp::SparsePattern, dtype::ForwardAD, func!, nx, ng)

Cache for sparse jacobian using ForwardDiff

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function sparsejacobiancache(sp::SparsePattern, dtype::ForwardAD, func!, nx, ng)

    g = zeros(ng)
    x = zeros(nx)
    Jsp = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colors = SparseDiffTools.matrix_colors(Jsp)
    cachesp = SparseDiffTools.ForwardColorJacCache(func!, x, dx=g, colorvec=colors, sparsity=Jsp)

    return GradOrJacCache(func!, Jsp, cachesp, dtype)
end


"""
    sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4} where {T1,T2,T3,T4<:ForwardAD})

evaluate sparse jacobian using ForwardDiff

# Arguments
- `dg::Vector{Float}`: constraint jacobian, modified in place
- `x::Vector{Float}`: design variables, input
- `cache::GradOrJacCache`: cache generated by `sparsejacobiancache`
"""
function sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4}
    where {T1,T2,T3,T4<:ForwardAD})
    
    SparseDiffTools.forwarddiff_color_jacobian!(cache.work, cache.f!, x, cache.cache)
    dg[:] = cache.work.nzval

    return nothing
end


# """
#     sparsejacobiancache(sp::SparsePattern, dtype::Zygote, func!, nx, ng)

# Cache for sparse jacobian using Zygote

# # Arguments
# - `func!::function`: function of form: f = func!(g, x)
# - `nx::Int`: number of design variables
# - `ng::Int`: number of constraints
# """
# function sparsejacobiancache(sp::SparsePattern, dtype::Zygote, func!, nx, ng)

#     g = zeros(ng)
#     x = zeros(nx)
#     Jsp = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
#     colors = SparseDiffTools.matrix_colors(Jsp, partition_by_rows=true)  # color by rows

#     # cachesp = SparseDiffTools.ForwardColorJacCache(func!, x, dx=g, colorvec=colors, sparsity=Jsp)

#     return GradOrJacCache(func!, Jsp, cachesp, dtype)
# end


# """
#     sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4} where {T1,T2,T3,T4<:Zygote})

# evaluate sparse jacobian using Zygote

# # Arguments
# - `dg::Vector{Float}`: constraint jacobian, modified in place
# - `x::Vector{Float}`: design variables, input
# - `cache::GradOrJacCache`: cache generated by `sparsejacobiancache`
# """
# function sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4}
#     where {T1,T2,T3,T4<:Zygote})
    
#     # SparseDiffTools.forwarddiff_color_jacobian!(cache.work, cache.f!, x, cache.cache)
#     # dg[:] = cache.work.nzval

#     return nothing
# end

"""
    sparsejacobiancache(sp::SparsePattern, dtype::ForwardAD, func!, nx, ng)

Cache for sparse jacobian using finite differencing

# Arguments
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function sparsejacobiancache(sp::SparsePattern, dtype::FD, func!, nx, ng)

    g = zeros(ng)
    x = zeros(nx)
    Jsp = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colors = SparseDiffTools.matrix_colors(Jsp)
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(x, fdtype, colorvec=colors, sparsity=Jsp)

    return GradOrJacCache(func!, Jsp, cache, dtype)
end


"""
    sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4} where {T1,T2,T3,T4<:FD})

evaluate sparse jacobian using finite differencing

# Arguments
- `dg::Vector{Float}`: constraint jacobian, modified in place
- `x::Vector{Float}`: design variables, input
- `cache::GradOrJacCache`: cache generated by `sparsejacobiancache`
"""
function sparsejacobian!(dg, x, cache::GradOrJacCache{T1,T2,T3,T4}
    where {T1,T2,T3,T4<:FD})

    FiniteDiff.finite_difference_jacobian!(cache.work, cache.f!, x, cache.cache)
    dg[:] = cache.work.nzval
    
    return nothing
end


# ----- gradient and sparse jacobian ----------

# internally used cache for gradients/jacobians
struct SparseCache{T1,T2}
    gradcache::T1
    jaccache::T2
end

# """
# same derivative method for gradient and jacobian
# """
# function createcache(sp::SparsePattern, dtype, func!, nx, ng)
#     gradcache = gradientcache(dtype, func!, nx, ng)
#     jaccache = sparsejacobiancache(sp, dtype, func!, nx, ng)

#     return SparseCache(gradcache, jaccache)
# end

"""
    createcache(sp::SparsePattern, dtype::T, func!, nx, ng) where T<:Vector

create cache for derivatives when the jacobian is sparse

# Arguments
- `sp::SparsePattern`: sparsity pattern
- `dtype::Vector{AbstractDiffMethod}`: differentiation method for gradient and jacobian (array of length two)
- `func!::function`: function of form: f = func!(g, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::SparsePattern, dtype::T, func!, nx, ng) where T<:Vector
    gradcache = gradientcache(dtype[1], func!, nx, ng)
    jaccache = sparsejacobiancache(sp, dtype[2], func!, nx, ng)

    return SparseCache(gradcache, jaccache)
end


"""
    evaluate!(g, df, dg, x, cache::T) where T <: SparseCache

evaluate function and derivatives for a sparse jacobian

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::SparseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::SparseCache{T1,T2}) where {T1,T2}
   
    f = cache.gradcache.f!(g, x)
    gradient!(df, x, cache.gradcache)
    sparsejacobian!(dg, x, cache.jaccache)

    return f
end


"""
    createcache(sp::SparsePattern, dtype::UserDeriv, func!, nx, ng)

Cache for sparse jacobian with user-supplied derivatives

# Arguments
- `func!::function`: function of form: f = func!(g, df, dg, x)
- `nx::Int`: number of design variables
- `ng::Int`: number of constraints
"""
function createcache(sp::SparsePattern, dtype::UserDeriv, func!, nx, ng)
    return GradOrJacCache(func!, 0.0, nothing, dtype)
end

"""
    evaluate!(g, df, dg, x, cache::DenseCache{T1,T2,T3,T4,T5} where {T1,T2,T3,T4,T5<:UserDeriv})

evaluate function and derivatives for a dense jacobian with user-provided derivatives

# Arguments
- `g::Vector{Float}`: constraints, modified in place
- `df::Vector{Float}`: objective gradient, modified in place
- `dg::Vector{Float}`: constraint jacobian, modified in place (order specified by sparsity pattern)
- `x::Vector{Float}`: design variables, input
- `cache::DenseCache`: cache generated by `createcache`
"""
function evaluate!(g, df, dg, x, cache::GradOrJacCache{T1,T2,T3,T4} 
    where {T1,T2,T3,T4<:UserDeriv})
    f = cache.f!(g, df, dg, x)
    
    return f
end

