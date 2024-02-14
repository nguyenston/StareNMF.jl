include("synthetic_data.jl")
include("stare_nmf.jl")

using Distributions
using LinearAlgebra
using NMF
using .SyntheticData
using .Util
using .StareNMF

using GLMakie
GLMakie.activate!()

function main()
  K = 4
  K_guess = 5
  diri = Dirichlet(2K, 1)
  W = rand(diri, K)
  H = 100 * rand(K, 100)
  X = make_count_matrix(W, H)

  empirical_eps = generate_empirical_eps_sets(X, W, H)
  display(sum(distance_from_standard_uniform.(empirical_eps); dims=1))

  result = nnmf(Float64.(X), K_guess; alg=:multdiv, maxiter=12000, replicates=1)
  empirical_eps = generate_empirical_eps_sets(X, result.W, result.H)
  display(sum(distance_from_standard_uniform.(empirical_eps); dims=1))

  # resultW = reduce(hcat, normalize!.(eachcol(result.W), 1))
  #
  # f = Figure(backgroundcolor=:tomato)
  # print("W = ")
  # display(W)
  # print("result.W = ")
  # display(resultW)
  # println("niters = ", result.niters)
  # println("converged = ", result.converged)
  # println("objvalue = ", result.objvalue)
  # wait(display(f))
end

main()
