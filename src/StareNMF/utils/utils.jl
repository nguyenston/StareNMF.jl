module Utils
export invcdf, threaded_nmf, count_matrix_from_WH
export signature_plot, signature_side2side, signature_bestmatch, bubbles, rho_k_losses
export compare_against_gt, BIC, sARI, rho_k_bottom
export rho_performance_factory

using NMF
using BSSMF
using Printf
using MultivariateStats
using Statistics
using Distributions

include("./plotutils.jl")
include("./mvcnmf.jl")
include("./nmfutils.jl")


"""
The generalized inverse of the cdf of a univariate measure
"""
function invcdf(d::UnivariateDistribution, lp::Real)
  @assert 0 <= lp <= 1 "lp needs to be between 0 an 1"
  return invlogcdf(d, log(lp))
end

"""
TODO: add documentation
"""
function count_matrix_from_WH(W::Matrix{T}, H::Matrix{T}) where {T<:Number}
  V = W * H
  distr = Poisson.(V)
  return rand.(distr)
end

"""
TODO: add documentation
"""
function BIC(X, nmf_result::NMF.Result{T}; model=(mu, sigma) -> Normal(mu, sigma), modelargs=(1,)) where {T}
  K, N = size(nmf_result.H)
  WH = nmf_result.W * nmf_result.H
  lpdf = logpdf.(model.(WH, modelargs...), X)
  return K * log(N) - 2sum(lpdf) + 2log(factorial(big(K)))
end

"""
TODO: add documentation
"""
function sARI(H1, H2)
  N = size(H1, 2)
  colnormalize = X -> reduce(hcat, normalize.(eachcol(X), 1))
  Prc = colnormalize(H1) * colnormalize(H2)'
  Pr = sum(Prc; dims=2)
  Pc = sum(Prc; dims=1)

  Prc_sqr = sum(Prc .^ 2)
  Pr_sqr = sum(Pr .^ 2)
  Pc_sqr = sum(Pc .^ 2)

  a = 0.5 * (Prc_sqr - N)
  b = 0.5 * (Pr_sqr - Prc_sqr)
  c = 0.5 * (Pc_sqr - Prc_sqr)
  d = 0.5 * (Prc_sqr + N^2 - Pr_sqr - Pc_sqr)

  Nc2 = a + b + c + d
  ERI = (a + b) * (a + c) + (c + d) * (b + d)
  nom = Nc2 * (a + d) - ERI
  denom = Nc2^2 - ERI

  return nom / denom
end
end
