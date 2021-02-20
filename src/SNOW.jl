module SNOW

using ForwardDiff
using ReverseDiff
using Zygote
using DiffResults
using FiniteDiff
using SparseArrays
using SparseDiffTools
using Snopt
using Ipopt


include("derivatives.jl")

export ForwardAD, ReverseAD, RevZyg, FD, UserDeriv
export DensePattern, SparsePattern

include("interface.jl")

export minimize
export Options

include("solvers.jl")

export SNOPT, IPOPT


end
