module SNOW

using ForwardDiff
using ReverseDiff
# using Zygote
using DiffResults
using FiniteDiff
using SparseArrays
using SparseDiffTools
using Requires
using Ipopt


include("derivatives.jl")

export ForwardAD, ReverseAD, RevZyg, ForwardFD, CentralFD, ComplexStep, UserDeriv
export DensePattern, SparsePattern

include("interface.jl")

export minimize
export Options

include("ipopt.jl")

export IPOPT

# conditionally load Snopt
function __init__()
    @require Snopt="0e9dc826-d618-11e8-1f57-c34e87fde2c0" include("snopt.jl")
end

end
