"""
A technique to determine the appropriate rank K of an NMF
"""
module StareNMF
include("./utils.jl")

export UniformApproximate
export PiecewiseUniform, KDEUniform
export distance_from_standard_uniform
export generate_empirical_eps_sets, structurally_aware_loss
export componentwise_loss, stare_from_componentwise_loss

using Distributions
using KernelDensity
using LinearAlgebra

abstract type UniformApproximate end

"""
Distribution represented by a series of bins with width and height
"""
struct PiecewiseUniform{T<:AbstractFloat} <: UniformApproximate
  data::Vector{Tuple{T,T}}
  ledge_list::Vector{Tuple{T,T}} # for logging
  components::Vector{Tuple{T,T}} # for logging

  PiecewiseUniform(components::Vector{Tuple{T,T}}; tol::T=1e-4) where {T} = begin
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
  KDEUniform(components::Vector{Tuple{T,T}}; multiplier::Integer=1, save_sample=false) where {T} = begin
    samples = T[]
    for (a, b) in components
      if !(0 <= a <= b <= 1)
        println("Warning: invalid component: a = $(a), b = $(b), skipped")
        continue
      end

      if a < b
        append!(samples, rand(Uniform(a, b), multiplier))
      else
        append!(samples, fill(a, multiplier))
      end
    end
    return new{T}(save_sample ? samples : T[], kde(samples, boundary=(0, 1), kernel=Epanechnikov))
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

function componentwise_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F};
  approx_type::Type{T}=KDEUniform, nysamples::Integer=1, approxargs=()) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  empirical_eps = generate_empirical_eps_sets(X, W, H, approx_type; nysamples, approxargs)
  return dropdims(sum(distance_from_standard_uniform.(empirical_eps); dims=1); dims=1)
end

function stare_from_componentwise_loss(cwl, rho; lambda=0.01)
  K = length(cwl)
  return sum(max.(0, cwl .- rho)) + lambda * K
end

function structurally_aware_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, rho::F;
  lambda::F=0.01, approx_type::Type{T}=KDEUniform,
  kwargs...) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  componentwise_loss = componentwise_loss(X, W, H; approx_type, kwargs)
  K = length(componentwise_loss)
  return sum(max.(0, componentwise_loss .- rho)) + lambda * K
end

function structurally_aware_loss(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, rho::Vector{F};
  lambda::F=0.01, approx_type::Type{T}=KDEUniform,
  kwargs...) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

  componentwise_loss = componentwise_loss(X, W, H; approx_type, kwargs)
  K = length(componentwise_loss)
  return [sum(max.(0, componentwise_loss .- rh)) + lambda * K for rh in rho]
end


function generate_empirical_eps_sets(X::Matrix{R}, W::Matrix{F}, H::Matrix{F}, approx_type::Type{T};
  nysamples::Integer=1, approxargs=()) where {T<:UniformApproximate,F<:AbstractFloat,R<:Real}

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
  for _ in 1:nysamples, n = 1:N
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
  return approx_type.(eps_conditional; approxargs...)
end

end
