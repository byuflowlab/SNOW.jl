using SNOW
using Test
using Zygote
using ForwardDiff
using SparseArrays

checkallocations = false
snopttest = false

@testset "derivatives" begin

function test1!(g, x)

    f = x[1]^2 - x[2]

    # Zygote.ignore() do
    g[1] = x[2] - 2*x[1]
    g[2] = -x[2]
    g[3] = x[1]^2
    # end
    return f
end

# ---- test sparsity pattern  --------
ng = 3
lx = [-5; -5]
ux = [5; 5]
sp = SparsePattern(ForwardAD(), test1!, ng, lx, ux)
@test sp.rows == [1, 3, 1, 2]
@test sp.cols == [1, 1, 2, 2]
# -------------------------------------

# ----- test derivatives ------------
nx = 2
ng = 3
g = zeros(ng)
df = zeros(nx)
dg = zeros(ng*nx)
x = [1.0, 2.0]

# forward AD
cache = SNOW.createcache(DensePattern(), ForwardAD(), test1!, nx, ng)
SNOW.evaluate!(g, df, dg, x, cache)

@test df == [2*x[1]; -1.0]
@test dg == [-2.0, 0.0, 2*x[1], 1.0, -1.0, 0.0]

# reverse AD
cache = SNOW.createcache(DensePattern(), ReverseAD(), test1!, nx, ng)
SNOW.evaluate!(g, df, dg, x, cache)
@test df == [2*x[1]; -1.0]
@test dg == [-2.0, 0.0, 2*x[1], 1.0, -1.0, 0.0]

# finite diff
cache = SNOW.createcache(DensePattern(), ForwardFD(), test1!, nx, ng)
SNOW.evaluate!(g, df, dg, x, cache)

@test isapprox(df, [2*x[1]; -1.0])
@test isapprox(dg, [-2.0, 0.0, 2*x[1], 1.0, -1.0, 0.0])

# sparse with reverse and forward
dg = zeros(length(sp.rows))
cache = SNOW.createcache(sp, [ReverseAD(), ForwardAD()], test1!, nx, ng)
SNOW.evaluate!(g, df, dg, x, cache)

@test df == [2*x[1]; -1.0]
@test dg == [-2.0, 2*x[1], 1.0, -1.0]

# # sparse with zygote and forward
# dg = zeros(length(sp.rows))
# cache = SNOW.createcache(sp, [RevZyg(), ForwardAD()], test1!, nx, ng)
# SNOW.evaluate!(g, df, dg, x, cache)

# @test df == [2*x[1]; -1.0]
# @test dg == [-2.0, 2*x[1], 1.0, -1.0]

# ----------------------------------------

end


@testset "sparse jacobian - finite differencing" begin

# also exercises a trailing all-zero row/column: g[end] doesn't depend on
# x at all, and x[end] doesn't affect any constraint. this previously
# crashed the ForwardAD sparse cache (see SNOW.jl PR #25 review) because
# the sparsity matrix was rebuilt without explicit (ng, nx) dimensions,
# so a shorter colorvec than nx was produced.
function dropped!(g, x)
    n = length(x)
    for i in 1:n-1
        g[i] = x[i]^2 + sin(x[i])
    end
    g[n] = 1.0
    return sum(abs2, x[1:n-1])
end

nx = 5
ng = 5
lx = -5*ones(nx)
ux = 5*ones(nx)
sp = SparsePattern(ForwardAD(), dropped!, ng, lx, ux)
x = collect(range(0.2, 1.8, length=nx))
Jdense = ForwardDiff.jacobian(dropped!, zeros(ng), x)

# forward AD sparse jacobian (regression test for the dimension bug)
cache = SNOW.sparsejacobiancache(sp, ForwardAD(), dropped!, nx, ng)
dg = zeros(length(sp.rows))
SNOW.sparsejacobian!(dg, x, cache)
Jsparse = Matrix(sparse(sp.rows, sp.cols, dg, ng, nx))
@test isapprox(Jsparse, Jdense; atol=1e-10)

# sparse jacobians via finite differencing (previously had no test coverage at all)
for dtype in (ForwardFD(), CentralFD(), ComplexStep())
    cache = SNOW.sparsejacobiancache(sp, dtype, dropped!, nx, ng)
    dgfd = zeros(length(sp.rows))
    SNOW.sparsejacobian!(dgfd, x, cache)
    Jsparsefd = Matrix(sparse(sp.rows, sp.cols, dgfd, ng, nx))
    @test isapprox(Jsparsefd, Jdense; atol=1e-4)
end

end


@testset "sparse jacobian - many colors" begin

# g[1] depends on every x[i], so every column conflicts with every other
# column in the coloring graph -> ncolors == nx. stresses the coloring path
# well beyond the single/few-color cases covered above.
function starfun!(g, x)
    n = length(x)
    g[1] = sum(x)
    for i in 2:n
        g[i] = x[i]^2 + 0.1*x[1]
    end
    return sum(abs2, x)
end

nx = 20
ng = 20
lx = -5*ones(nx)
ux = 5*ones(nx)
sp = SparsePattern(ForwardAD(), starfun!, ng, lx, ux)
x = collect(range(0.1, 2.0, length=nx))
Jdense = ForwardDiff.jacobian(starfun!, zeros(ng), x)

cache = SNOW.sparsejacobiancache(sp, ForwardAD(), starfun!, nx, ng)
dg = zeros(length(sp.rows))
SNOW.sparsejacobian!(dg, x, cache)
Jsparse = Matrix(sparse(sp.rows, sp.cols, dg, ng, nx))
@test isapprox(Jsparse, Jdense; atol=1e-8)

end


@testset "sparse jacobian - non-square" begin

# more constraints than variables (overdetermined), banded + wraparound
# coupling so ng != nx and the sparsity matrix is rectangular
function overdetermined!(g, x)
    nx = length(x)
    ng = length(g)
    for i in 1:ng
        j1 = ((i - 1) % nx) + 1
        j2 = (i % nx) + 1
        g[i] = 0.3*x[j1]^2 + (j2 != j1 ? 0.2*x[j2] : 0.0)
    end
    return sum(abs2, x)
end

nx = 6
ng = 14
lx = -5*ones(nx)
ux = 5*ones(nx)
x = collect(range(0.15, 1.9, length=nx))
Jdense = ForwardDiff.jacobian(overdetermined!, zeros(ng), x)

sp = SparsePattern(ForwardAD(), overdetermined!, ng, lx, ux)
@test size(sparse(sp.rows, sp.cols, ones(length(sp.rows)), ng, nx)) == (ng, nx)

cache = SNOW.sparsejacobiancache(sp, ForwardAD(), overdetermined!, nx, ng)
dg = zeros(length(sp.rows))
SNOW.sparsejacobian!(dg, x, cache)
Jsparse = Matrix(sparse(sp.rows, sp.cols, dg, ng, nx))
@test isapprox(Jsparse, Jdense; atol=1e-8)

for dtype in (ForwardFD(), CentralFD(), ComplexStep())
    cachefd = SNOW.sparsejacobiancache(sp, dtype, overdetermined!, nx, ng)
    dgfd = zeros(length(sp.rows))
    SNOW.sparsejacobian!(dgfd, x, cachefd)
    Jsparsefd = Matrix(sparse(sp.rows, sp.cols, dgfd, ng, nx))
    @test isapprox(Jsparsefd, Jdense; atol=1e-4)
end

end


@testset "sparse jacobian - repeated evaluation" begin

# a single cache must produce correct results across many calls at
# different x, not just the first call (catches stale-state bugs in
# reused scratch buffers / prepared coloring state)
function tridiagonal!(g, x)
    n = length(x)
    g[1] = x[1]^2 - 2*x[2]
    for i in 2:n-1
        g[i] = x[i-1]*x[i] - x[i+1]^2 + sin(x[i])
    end
    g[n] = x[n]^2 - x[n-1]
    return sum(abs2, x)
end

nx = 10
ng = 10
lx = -5*ones(nx)
ux = 5*ones(nx)
sp = SparsePattern(ForwardAD(), tridiagonal!, ng, lx, ux)

cache = SNOW.sparsejacobiancache(sp, ForwardAD(), tridiagonal!, nx, ng)
dg = zeros(length(sp.rows))

for trial in 1:5
    x = collect(range(0.1*trial, 0.1*trial + 1.5, length=nx))
    SNOW.sparsejacobian!(dg, x, cache)
    Jsparse = Matrix(sparse(sp.rows, sp.cols, dg, ng, nx))
    Jdense = ForwardDiff.jacobian(tridiagonal!, zeros(ng), x)
    @test isapprox(Jsparse, Jdense; atol=1e-8)
end

end


@testset "sparse jacobian - empty sparsity" begin

# constraints that don't depend on x at all: sp.rows/cols come back empty.
# the old manual SparseDiffTools-replacement code had an explicit ncolor==0
# branch for this; confirm the DifferentiationInterface path also degrades
# gracefully instead of erroring.
function constantfun!(g, x)
    g[1] = 5.0
    g[2] = -3.0
    return sum(abs2, x)
end

nx = 4
ng = 2
lx = -5*ones(nx)
ux = 5*ones(nx)
sp = SparsePattern(ForwardAD(), constantfun!, ng, lx, ux)
@test isempty(sp.rows)

x = collect(range(0.3, 1.2, length=nx))

cache = SNOW.sparsejacobiancache(sp, ForwardAD(), constantfun!, nx, ng)
dg = zeros(length(sp.rows))
SNOW.sparsejacobian!(dg, x, cache)
@test isempty(dg)

for dtype in (ForwardFD(), CentralFD(), ComplexStep())
    cachefd = SNOW.sparsejacobiancache(sp, dtype, constantfun!, nx, ng)
    dgfd = zeros(length(sp.rows))
    SNOW.sparsejacobian!(dgfd, x, cachefd)
    @test isempty(dgfd)
end

end

# if checkallocations
# # -------- test allocations --------------
#     using BenchmarkTools

#     function test2!(g, x)
#         f = x[1]^2 - x[2]
        
#         Zygote.ignore() do
#             nx = length(x)
#             for i = 1:nx
#                 g[i] = x[i]^2
#             end
#             g[nx+1:end] .= x[1]
#         end
#         return f
#     end

#     nx = 300
#     ng = 500
#     g = zeros(ng)
#     df = zeros(nx)
#     dg = zeros(ng*nx)
#     x = 2*ones(nx)
#     cache = SNOW.createcache(DensePattern(), ForwardAD(), test2!, nx, ng)
#     @btime SNOW.evaluate!($g, $df, $dg, $x, $cache)
#     # 577.123 μs (6 allocations: 2.30 MiB)

#     cache = SNOW.createcache(DensePattern(), ReverseAD(), test2!, nx, ng)
#     @btime SNOW.evaluate!($g, $df, $dg, $x, $cache)
#     # 4.188 ms (6 allocations: 2.30 MiB)

#     cache = SNOW.createcache(DensePattern(), FD("forward"), test2!, nx, ng)
#     @btime SNOW.evaluate!($g, $df, $dg, $x, $cache)
#     # 358.457 μs (6 allocations: 2.30 MiB)

#     lx = -5*ones(nx)
#     ux = 5*ones(nx)
#     sp = SparsePattern(ForwardAD(), test2!, ng, lx, ux)
#     dg = zeros(length(sp.rows))
#     cache = SNOW.createcache(sp, [ReverseAD(), ForwardAD()], test2!, nx, ng)
#     @btime SNOW.evaluate!($g, $df, $dg, $x, $cache)
#     # 17.391 μs (0 allocations: 0 bytes)

#     cache = SNOW.createcache(sp, [RevZyg(), ForwardAD()], test2!, nx, ng)
#     @btime SNOW.evaluate!($g, $df, $dg, $x, $cache)
#     # 9.865 μs (6 allocations: 11.63 KiB)
# end
# # ----------------------------------------


@testset "optimization" begin

function barnes(g, x)

    a1 = 75.196
    a3 = 0.12694
    a5 = 1.0345e-5
    a7 = 0.030234
    a9 = 3.5256e-5
    a11 = 0.25645
    a13 = 1.3514e-5
    a15 = -5.2375e-6
    a17 = 7.0e-10
    a19 = -1.6638e-6
    a21 = 0.0005
    a2 = -3.8112
    a4 = -2.0567e-3
    a6 = -6.8306
    a8 = -1.28134e-3
    a10 = -2.266e-7
    a12 = -3.4604e-3
    a14 = -28.106
    a16 = -6.3e-8
    a18 = 3.4054e-4
    a20 = -2.8673

    x1 = x[1]
    x2 = x[2]
    y1 = x1*x2
    y2 = y1*x1
    y3 = x2^2
    y4 = x1^2

    # --- function value ---

    f = a1 + a2*x1 + a3*y4 + a4*y4*x1 + a5*y4^2 +
        a6*x2 + a7*y1 + a8*x1*y1 + a9*y1*y4 + a10*y2*y4 +
        a11*y3 + a12*x2*y3 + a13*y3^2 + a14/(x2+1) +
        a15*y3*y4 + a16*y1*y4*x2 + a17*y1*y3*y4 + a18*x1*y3 +
        a19*y1*y3 + a20*exp(a21*y1)

    # --- constraints ---

    g[1] = 1 - y1/700.0
    g[2] = y4/25.0^2 - x2/5.0
    g[3] = (x1/500.0- 0.11) - (x2/50.0-1)^2

    return f  
end


x0 = [10.0; 10.0]
lx = [0.0; 0.0]
ux = [65.0; 70.0]

ng = 3
options = Options(solver=IPOPT(), derivatives=ForwardFD())
xopt, fopt, info, out = minimize(barnes, x0, ng, lx, ux, -Inf, 0.0, options)


@test isapprox(xopt[1], 49.5263; atol=1e-4)
@test isapprox(xopt[2], 19.6228; atol=1e-4)
@test isapprox(fopt, -31.6368; atol=1e-4)
@test info == :Solve_Succeeded || info == :Solved_To_Acceptable_Level

# same problem, but through a SparsePattern + DifferentiationInterface-based
# ForwardAD sparse jacobian, end-to-end through the real Ipopt callback loop
# (Ipopt calls the jacobian callback many times at different x during the
# solve, unlike the isolated single/few-call tests above)
sp_barnes = SparsePattern(ForwardAD(), barnes, ng, lx, ux)
options = Options(solver=IPOPT(), derivatives=[ReverseAD(), ForwardAD()], sparsity=sp_barnes)
xopt, fopt, info, out = minimize(barnes, x0, ng, lx, ux, -Inf, 0.0, options)

@test isapprox(xopt[1], 49.5263; atol=1e-4)
@test isapprox(xopt[2], 19.6228; atol=1e-4)
@test isapprox(fopt, -31.6368; atol=1e-4)
@test info == :Solve_Succeeded || info == :Solved_To_Acceptable_Level

if snopttest
    options = Options(solver=SNOPT(), derivatives=ForwardAD())
    xopt, fopt, info, out = minimize(barnes, x0, ng, lx, ux, -Inf, 0.0, options)

    @test isapprox(xopt[1], 49.5263; atol=1e-4)
    @test isapprox(xopt[2], 19.6228; atol=1e-4)
    @test isapprox(fopt, -31.6368; atol=1e-4)
    @test info == "Finished successfully: optimality conditions satisfied"

    options = Options(solver=SNOPT(), derivatives=ComplexStep())
    xopt, fopt, info, out = minimize(barnes, x0, ng, lx, ux, -Inf, 0.0, options)

    @test isapprox(xopt[1], 49.5263; atol=1e-4)
    @test isapprox(xopt[2], 19.6228; atol=1e-4)
    @test isapprox(fopt, -31.6368; atol=1e-4)
    @test info == "Finished successfully: optimality conditions satisfied"
end


function sparsegrad(g, x)

    f = x[1]^2 - x[2]

    g[1] = x[2] - 2*x[1]
    g[2] = -x[2]

    return f
end

x0 = [0.0; 0.0]
lx = [-10.0, -10.0]
ux = [10.0, 10.0]
ng = 2

# detect sparsity pattern
sp = SparsePattern(ForwardAD(), sparsegrad, ng, lx, ux)

if snopttest
    options = Options(solver=SNOPT(), derivatives=[ReverseAD(), ForwardAD()], sparsity=sp)
    xopt, fopt, info, out = minimize(sparsegrad, x0, ng, lx, ux, -Inf, 0.0, options)

    @test isapprox(xopt[1], 1.0; atol=1e-6)
    @test isapprox(xopt[2], 2.0; atol=1e-6)
    @test isapprox(fopt, -1.0; atol=1e-5)
    @test info == "Finished successfully: optimality conditions satisfied"
end

end


# @testset "problem format" begin

# function test3!(g, prob, x)
#     return nothing
# end

# prob = SNOW.createproblem(test3!, "test3")

# x = [1.0, 2.0, 3.0]
# lx = [0.0, 0.0, 0.0]
# ux = 10*ones(3)
# names = ["c1", "c2", "c3"]
# SNOW.add_dv!(prob, x, lx, ux, names)

# x2 = [8.0, 2.0]
# lx2 = [0.0, 0.0, 0.0]
# ux2 = 5*ones(3)
# names2 = ["t1", "t2"]
# SNOW.add_dv!(prob, x2, lx2, ux2, names2)

# x = rand(5)
# SNOW.get_dvs(prob, x)

# end