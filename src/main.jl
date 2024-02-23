include("StareNMF/StareNMF.jl")

using Distributions
using LinearAlgebra
using NMF
using .StareNMF
using .StareNMF.Utils

using GLMakie
using CairoMakie
CairoMakie.activate!(type="svg")

function main()
  K = 5
  diri = Dirichlet(4K, 0.1)
  W = rand(diri, K)
  H = 100 * rand(K, 100)
  X = count_matrix_from_WH(W, H)

  empirical_eps = generate_empirical_eps_sets(X, W, H, PiecewiseUniform)
  componentwise_loss = sum(distance_from_standard_uniform.(empirical_eps); dims=1)
  display(componentwise_loss)
  println("W = ")
  display(W)
  println("Gram matrix of W:")
  W_l2 = W * Diagonal(1 ./ norm.(eachcol(W)))
  display(W_l2' * W_l2)

  rho = collect(0:0.01:1.5)
  losses_kde = Vector{Float64}[]
  for k in 1:8
    result = nnmf(Float64.(X), k; alg=:multdiv, maxiter=30000, replicates=1)
    push!(losses_kde, structurally_aware_loss(X, result.W, result.H, rho=rho, lambda=0.01, approx_type=KDEUniform))
  end

  losses_piecewise = Vector{Float64}[]
  for k in 1:8
    result = nnmf(Float64.(X), k; alg=:multdiv, maxiter=30000, replicates=1)
    push!(losses_piecewise, structurally_aware_loss(X, result.W, result.H, rho=rho, lambda=0.01, approx_type=PiecewiseUniform))
  end

  fig = Figure(size=(800, 1200))
  ax1 = Axis(fig[1, 1], yscale=log10, title="K=$(K), D=$(4K), KDE")
  for k in 1:8
    lines!(ax1, rho, losses_kde[k], label="k=$(k)")
  end
  ax2 = Axis(fig[2, 1], yscale=log10, title="K=$(K), D=$(4K), Piecewise")
  for k in 1:8
    lines!(ax2, rho, losses_piecewise[k], label="k=$(k)")
  end
  Legend(fig[1, 2], ax1, "Legend")
  Legend(fig[2, 2], ax2, "Legend")
  save("plot.svg", fig)
  # wait(display(fig))
end

main()
