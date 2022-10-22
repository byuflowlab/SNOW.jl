module SNOW

import ForwardDiff
# using Zygote
using DiffResults
import FiniteDiff
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
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("ReverseAD.jl")
end

end
