module Utils
export invcdf, threaded_nmf, count_matrix_from_WH
export signature_plot, signature_side2side, signature_bestmatch, bubbles, rho_k_losses
export compare_against_gt, BIC, rho_k_bottom
export rho_performance_factory

using NMF
using BSSMF
using Printf
using MultivariateStats
using Statistics

include("./plotutils.jl")
include("./nmfutils.jl")
include("./mvcnmf.jl")

using Distributions

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


end
