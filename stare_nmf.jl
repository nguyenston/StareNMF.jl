"""
Variouse utilities functions
"""
module Util
export invcdf, distance_from_standard_uniform
export PiecewiseUniform

using Distributions

"""
"""
mutable struct PiecewiseUniform{T<:AbstractFloat}
  data::Vector{Tuple{T,T}}
  PiecewiseUniform{T}(components::Vector{Tuple{T,T}}) where {T} = begin
    n = length(components)
    ledge_list = []
    for (a, b) in components
      @assert 0 <= a <= b <= 1 "invalid component: a = $(a), b = $(b)"
      px = 1 / (b - a) / n
      push!(ledge_list, (a, px))
      push!(ledge_list, (b, -px))
    end
    sort!(ledge_list; by=first)

    height = 0
    data = []
    for i = 1:length(ledge_list)-1
      height += ledge_list[i][2]
      if abs(height) < 1e-5
        height = max(0, height)
      end
      width = ledge_list[i+1][1] - ledge_list[i][1]
      push!(data, (width, height))
    end
    return new{T}(data)
  end
end

"""
Compute the KL Divergence D_KL(d|Uniform(0,1))
"""
function distance_from_standard_uniform(d::PiecewiseUniform)
  loss_map = ((w, h),) -> w * h * log(h)
  losses = loss_map.(d.data)
  valid_losses = losses[.!isnan.(losses)]
  return sum(valid_losses)
end

"""
The generalized inverse of the cdf of a univariate measure
"""
function invcdf(d::UnivariateDistribution, lp::Real)
  @assert 0 <= lp <= 1 "lp needs to be between 0 an 1"
  return invlogcdf(d, log(lp))
end
end

"""
A technique to determine the appropriate rank K of an NMF
"""
module StareNMF
export generate_empirical_eps_sets

using LinearAlgebra
using Distributions
using ..Util

function structurally_aware_loss(X, W, H; rho, lambda)
  empirical_eps = generate_empirical_eps_sets(X, W, H)
  componentwise_loss = sum(distance_from_standard_uniform.(empirical_eps); dims=1)
end

function generate_empirical_eps_sets(X, W, H)
  # sanity checking
  D_X, N_X = size(X)
  D_W, K_W = size(W)
  K_H, N_H = size(H)
  @assert D_X == D_W "Dimension mismatch between X, W, H"
  @assert K_W == K_H "Dimension mismatch between X, W, H"
  @assert N_X == N_H "Dimension mismatch between X, W, H"

  D = D_X
  K = K_W
  N = N_X

  eps_conditional = reshape([Tuple{Float64,Float64}[] for _ in 1:D*K], (D, K))

  y_to_eps_conditional = (y, Wdh) -> (cdf(Poisson(Wdh), y - 1), cdf(Poisson(Wdh), y))
  for n = 1:N
    x = X[:, n]
    h = H[:, n]
    Wdh = W * Diagonal(h)
    y_dist = Multinomial.(x, normalize.(eachrow(Wdh), 1))
    y = reduce(hcat, rand.(y_dist)) |> transpose # DxK matrix
    push!.(eps_conditional, y_to_eps_conditional.(y, Wdh))
  end
  return PiecewiseUniform{Float64}.(eps_conditional)
end

end
