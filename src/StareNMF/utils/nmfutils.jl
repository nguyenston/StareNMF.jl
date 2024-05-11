using NMF
using BSSMF

function run_nmf(X::AbstractMatrix{T}, k::Integer;
  init::Symbol=:nndsvdar,
  initdata=nothing,
  alg::Symbol=:greedycd,
  maxiter::Integer=100,
  tol::Real=cbrt(eps(T) / 100),
  W0::Union{AbstractMatrix{T},Nothing}=nothing,
  H0::Union{AbstractMatrix{T},Nothing}=nothing,
  update_H::Bool=true,
  verbose::Bool=false,
  kwargs...) where {T}
  eltype(X) <: Number && all(t -> t >= zero(T), X) || throw(ArgumentError("The elements of X must be non-negative."))

  p, n = size(X)
  k <= min(p, n) || throw(ArgumentError("The value of k should not exceed min(size(X))."))

  if !update_H && init != :custom
    @warn "Only W will be updated."
  end

  if init == :custom
    W0 !== nothing && H0 !== nothing || throw(ArgumentError("To use :custom initialization, set W0 and H0."))
    eltype(W0) <: Number && all(t -> t >= zero(T), W0) || throw(ArgumentError("The elements of W0 must be non-negative."))
    p0, k0 = size(W0)
    p == p0 && k == k0 || throw(ArgumentError("Invalid size for W0."))
    eltype(H0) <: Number && all(t -> t >= zero(T), H0) || throw(ArgumentError("The elements of H0 must be non-negative."))
    k0, n0 = size(H0)
    k == k0 && n == n0 || throw(ArgumentError("Invalid size for H0."))
  else
    W0 === nothing && H0 === nothing || @warn "Ignore W0 and H0 except for :custom initialization."
  end

  # determine whether H needs to be initialized
  initH = alg != :projals

  # perform initialization
  if init == :random
    W, H = NMF.randinit(X, k; zeroh=!initH, normalize=true)
  elseif init == :nndsvd
    W, H = NMF.nndsvd(X, k; zeroh=!initH, initdata=initdata)
  elseif init == :nndsvda
    W, H = NMF.nndsvd(X, k; variant=:a, zeroh=!initH, initdata=initdata)
  elseif init == :nndsvdar
    W, H = NMF.nndsvd(X, k; variant=:ar, zeroh=!initH, initdata=initdata)
  elseif init == :spa
    W, H = NMF.spa(X, k)
  elseif init == :custom
    W, H = W0, H0
  else
    throw(ArgumentError("Invalid value for init."))
  end

  # choose algorithm
  if alg == :projals
    alginst = NMF.ProjectedALS{T}(; maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :alspgrad
    alginst = NMF.ALSPGrad{T}(; maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :multmse
    alginst = NMF.MultUpdate{T}(; obj=:mse, maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :multdiv
    alginst = NMF.MultUpdate{T}(; obj=:div, maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :cd
    alginst = NMF.CoordinateDescent{T}(; maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :greedycd
    alginst = NMF.GreedyCD{T}(; maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :spa
    if init != :spa
      throw(ArgumentError("Invalid value for init, use :spa instead."))
    end
    alginst = NMF.SPA{T}(obj=:mse, kwargs...)
  else
    throw(ArgumentError("Invalid algorithm."))
  end
  return NMF.solve!(alginst, X, W, H)
end

"""
NMF.jl package's high level function nnmf, 
but can specify how many cpus to run in parallel for replicates.
If the algorithm name :bssmf, then we use the implementation 
on gitlab.com/vuthanho/BSSMF.jl instead. 
Note: ncpu is irrelevant for :bssmf
"""
function threaded_nmf(X::AbstractMatrix{T}, k::Integer;
  replicates::Integer=1, ncpu::Integer=1,
  alg::Symbol=:multdiv, init::Symbol=:nndsvdar,
  simplex_W::Bool=false, kwargs...) where {T}

  results = Vector{NMF.Result{Float64}}(undef, replicates)
  c = Channel() do ch
    foreach(i -> put!(ch, i), 1:replicates)
  end

  if alg == :bssmf
    l1_col_X = simplex_W ? norm.(eachcol(X), 1) : ones(Float64, size(X, 2))
    X_processed = simplex_W ? reduce(hcat, normalize.(eachcol(X), 1)) : X


    for i in eachindex(results)
      workspace = Workspace(X_processed, k)
      err, _ = bssmf!(workspace; kwargs...)

      l1_col_W = simplex_W ? norm.(eachcol(workspace.W)) : ones(Float64, size(workspace.W, 2))
      W = workspace.W ./ l1_col_W'
      H = workspace.H .* (l1_col_W * l1_col_X')
      results[i] = NMF.Result{Float64}(W, H, 0, true, err[end])
    end
  else
    Threads.foreach(c; ntasks=ncpu) do i
      _init = i == 1 ? init : :random
      results[i] = run_nmf(X, k; alg, init=_init, kwargs...)
    end
  end
  _, min_idx = findmin(x -> x.objvalue, results)
  return results[min_idx]
end

