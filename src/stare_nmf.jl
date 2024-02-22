"""
Variouse utilities functions
"""
module Util
export invcdf, distance_from_standard_uniform
export UniformApproximate
export PiecewiseUniform, KDEUniform

using Distributions
using KernelDensity
using Makie

abstract type UniformApproximate end

"""
Distribution represented by a series of bins with width and height
"""
struct PiecewiseUniform{T<:AbstractFloat} <: UniformApproximate
  data::Vector{Tuple{T,T}}
  ledge_list::Vector{Tuple{T,T}} # for logging
  components::Vector{Tuple{T,T}} # for logging

  PiecewiseUniform(components::Vector{Tuple{T,T}}; kwargs...) where {T} = begin
    tol = haskey(kwargs, :tol) ? convert(T, kwargs[:tol]) : convert(T, 1e-4)
    n = length(components)
    ledge_list = []
    for (a, b) in components
      # sanity guards
      @assert 0 <= a <= b <= 1 "invalid component: a = $(a), b = $(b)"
      if a == b
        continue
      end

      px = 1 / (b - a) / n
      push!(ledge_list, (a, px))
      push!(ledge_list, (b, -px))
    end
    sort!(ledge_list; by=first)

    height = 0
    data = []
    for i = 1:length(ledge_list)-1
      height += ledge_list[i][2]
      if abs(height) < tol
        height = max(0, height)
      end
      width = ledge_list[i+1][1] - ledge_list[i][1]
      push!(data, (width, height))
    end
    return new{T}(data, ledge_list, components)
  end
end

"""
Distribution represented by a kernel density estimation over sample points
"""
struct KDEUniform{T<:AbstractFloat} <: UniformApproximate
  samples::Vector{T}
  estimated_distr::UnivariateKDE
  KDEUniform(components::Vector{Tuple{T,T}}; kwargs...) where {T} = begin
    multiplier = haskey(kwargs, :multiplier) ? Int(kwargs[:multiplier]) : 1
    samples = T[]
    for (a, b) in components
      @assert 0 <= a <= b <= 1 "invalid component: a = $(a), b = $(b)"
      if a < b
        append!(samples, rand(Uniform(a, b), multiplier))
      else
        append!(samples, fill(a, multiplier))
      end
    end
    return new{T}(samples, kde(samples, boundary=(0, 1), kernel=Epanechnikov))
  end
end

"""
Compute the KL Divergence D_KL(d|Uniform(0,1))
"""
function distance_from_standard_uniform(d::PiecewiseUniform)
  function loss_map((w, h))
    @assert h >= 0 "negative h: $(h)"
    w * h * log(h)
  end
  losses = loss_map.(d.data)
  valid_losses = losses[.!isnan.(losses)]
  return sum(valid_losses)
end

"""
Compute the KL Divergence D_KL(d|Uniform(0,1))
"""
function distance_from_standard_uniform(d::KDEUniform)
  distr = d.estimated_distr
  width = step(distr.x)
  density = distr.density
  losses = width .* density .* log.(density)
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

"""
use makie to plot a mutation signature
"""
function plot_signature(gridpos, signatures, sig, title)
  s = signatures[:, sig]
  subfig = GridLayout()

  colors = [:blue, :black, :red, :grey, :green, :pink]
  barlabels = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
  color_index = collect(0:96-1) ./ 16 .|> floor .|> Int .|> x -> x + 1

  signature_axis = Axis(gridpos, xticksvisible=false, xticklabelsvisible=false, title=title,
    ytickformat=values -> ["$(round(v*100; digits=2))%" for v in values], limits=((-1, 97), (0, nothing)))
  label_axis = Axis(gridpos, xticks=(8:16:88, barlabels), xticksvisible=false,
    yticksvisible=false, yticklabelsvisible=false, xgridvisible=false, ygridvisible=false,
    bottomspinevisible=false, topspinevisible=false, leftspinevisible=false, rightspinevisible=false,
    xticklabelfont=:bold, limits=((-1, 97), (0, 1)))

  barplot!(signature_axis, 0.5:95.5, s, color=colors[color_index], width=1)
  barplot!(label_axis, 8:16:88, fill(1, 6); width=16, gap=0.05, color=colors,
    bar_labels=barlabels, label_offset=4, label_font=:bold)

  subfig[1:2, 1] = [signature_axis, label_axis]
  rowsize!(subfig, 1, Aspect(1, 0.25))
  rowsize!(subfig, 2, Aspect(1, 0.25 / 30))
  rowgap!(subfig, 2)
  gridpos[] = subfig
end
end

"""
A technique to determine the appropriate rank K of an NMF
"""
module StareNMF
export generate_empirical_eps_sets, structurally_aware_loss

using LinearAlgebra
using Distributions
using ..Util

function structurally_aware_loss(X, W, H; rho::Float64, lambda, approx_type::Type{T}, kwargs...) where {T<:UniformApproximate}
  empirical_eps = generate_empirical_eps_sets(X, W, H, approx_type; kwargs...)

  K = size(W)[2]
  componentwise_loss = sum(distance_from_standard_uniform.(empirical_eps); dims=1)
  return sum(max.(0, componentwise_loss .- rho)) + lambda * K
end

function structurally_aware_loss(X, W, H; rho::Vector{Float64}, lambda, approx_type::Type{T}, kwargs...) where {T<:UniformApproximate}
  empirical_eps = generate_empirical_eps_sets(X, W, H, approx_type; kwargs...)

  K = size(W)[2]
  componentwise_loss = sum(distance_from_standard_uniform.(empirical_eps); dims=1)
  return [sum(max.(0, componentwise_loss .- rh)) + lambda * K for rh in rho]
end

function generate_empirical_eps_sets(X, W, H, approx_type::Type{T}; kwargs...) where {T<:UniformApproximate}
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
    sanity_check = v -> if isnan(v[1])
      return ones(length(v)) / length(v)
    else
      return v
    end
    y_dist = Multinomial.(x, sanity_check.(normalize.(eachrow(Wdh), 1)))
    y = reduce(hcat, rand.(y_dist)) |> transpose # DxK matrix
    push!.(eps_conditional, y_to_eps_conditional.(y, Wdh))
  end
  return approx_type.(eps_conditional; kwargs...)
end

end
