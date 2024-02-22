"""
This module handle the generation of synthetic data, and maybe some other related utilities
"""
module SyntheticData
export make_count_matrix
using Distributions

function make_count_matrix(W::Matrix{T}, H::Matrix{T}) where {T<:Number}
  V = W * H
  distr = Poisson.(V)
  return rand.(distr)
end

end
