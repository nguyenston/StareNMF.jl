"""
Projection of vector onto the L1 simplex
"""
function condatProj!(y::AbstractArray{T,1}, epsilon::T=zero(T)) where {T<:AbstractFloat}
  a = 1
  N = size(y, 1)
  vtilde = Array{Int}(undef, N)
  v = Array{Int}(undef, N)
  v[1] = 1
  vlength = 1
  vtildelength = 0
  rho = y[1] - a
  for n in 2:N
    if y[n] > rho
      rho += (y[n] - rho) / (vlength + 1)
      if rho > y[n] - a
        vlength += 1
        v[vlength] = n
      else
        vtilde[vtildelength+1:vtildelength+vlength] = v[1:vlength]
        vtildelength += vlength
        v[1] = n
        vlength = 1
        rho = y[n] - a
      end
    end
  end
  if vtildelength != 0
    for i in 1:vtildelength
      yv = y[vtilde[i]]
      if yv > rho
        vlength += 1
        v[vlength] = vtilde[i]
        rho += (yv - rho) / vlength
      end
    end
  end
  vlengthold = vlength - 1
  while vlengthold != vlength
    vlengthold = vlength
    i = 1
    while i <= vlength
      yv = y[v[i]]
      if yv <= rho
        v[i] = v[vlength]
        vlength -= 1
        rho += (rho - yv) / vlength
      else
        i += 1
      end
    end
  end
  for i in 1:N
    yimrho = y[i] - rho
    if yimrho < epsilon
      y[i] = epsilon
    else
      y[i] = yimrho
    end
  end
  return nothing
end;

function nmf_skeleton!(updater::NMF.NMFUpdater{T},
  X, W::Matrix{T}, H::Matrix{T},
  maxiter::Int, verbose::Bool, tol, simplex_H) where {T}
  objv = convert(T, NaN)

  # init
  state = prepare_state(updater, X, W, H)
  preW = Matrix{T}(undef, size(W))
  preH = Matrix{T}(undef, size(H))
  if verbose
    start = time()
    objv = evaluate_objv(updater, state, X, W, H)
    @printf("%-5s    %-13s    %-13s    %-13s    %-13s\n", "Iter", "Elapsed time", "objv", "objv.change", "(W & H).change")
    @printf("%5d    %13.6e    %13.6e\n", 0, 0.0, objv)
  end

  # main loop
  converged = false
  t = 0
  if simplex_H
    condatProj!(H)
  end

  while !converged && t < maxiter
    t += 1
    copyto!(preW, W)
    copyto!(preH, H)

    # update H
    update_wh!(updater, state, X, W, H)
    if simplex_H
      condatProj!(H)
    end

    # determine convergence
    converged = stop_condition(W, preW, H, preH, tol)

    # display info
    if verbose
      elapsed = time() - start
      preobjv = objv
      objv = evaluate_objv(updater, state, X, W, H)
      @printf("%5d    %13.6e    %13.6e    %13.6e\n",
        t, elapsed, objv, objv - preobjv)
    end
  end

  if !verbose
    objv = evaluate_objv(updater, state, X, W, H)
  end
  return NMF.Result{T}(W, H, t, converged, objv)
end

# generic implementation
prepare_state(upd::NMF.NMFUpdater{T}, X, W, H) where {T} = NMF.prepare_state(upd, X, W, H)
update_wh!(upd::NMF.NMFUpdater{T}, state, X, W, H) where {T} = NMF.update_wh!(upd, state, X, W, H)
evaluate_objv(upd::NMF.NMFUpdater{T}, state, X, W, H) where {T} = NMF.evaluate_objv(upd, state, X, W, H)

function stop_condition(W::AbstractArray{T}, preW::AbstractArray, H::AbstractArray, preH::AbstractArray, eps::AbstractFloat) where {T}
  for j in axes(W, 2)
    dev_w = sum_w = zero(T)
    for i in axes(W, 1)
      dev_w += (W[i, j] - preW[i, j])^2
      sum_w += (W[i, j] + preW[i, j])^2
    end
    dev_h = sum_h = zero(T)
    for i in axes(H, 2)
      dev_h += (H[j, i] - preH[j, i])^2
      sum_h += (H[j, i] + preH[j, i])^2
    end
    if sqrt(dev_w) > eps * sqrt(sum_w) || sqrt(dev_h) > eps * sqrt(sum_h)
      return false
    end
  end
  return true
end

custom_solve!(alg::NMF.GreedyCD{T}, X, W, H, simplex_H=false) where {T} =
  nmf_skeleton!(NMF.GreedyCDUpd{T}(alg.update_H, alg.lambda_w, alg.lambda_h), X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)

custom_solve!(alg::NMF.ALSPGrad, X, W, H, simplex_H=false) =
  nmf_skeleton!(NMF.ALSPGradUpd(alg.update_H, alg.maxsubiter, alg.tolg), X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)

custom_solve!(alg::NMF.CoordinateDescent{T}, X, W, H, simplex_H=false) where {T} =
  nmf_skeleton!(NMF.CoordinateDescentUpd{T}(alg.α, alg.l₁ratio, alg.regularization, alg.shuffle, alg.update_H),
    X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)

custom_solve!(alg::MinVolConstrained{T}, X, W, H, simplex_H=false) where {T} = begin
  BSSMF.condatProj!(H)
  nmf_skeleton!(MinVolConstrainedUpd(alg.eta, alg.eta_decay, alg.eta_tol, alg.delta, alg.lambda, alg.maxsubiter, alg.update_H),
    X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)
end

# generic implementation
custom_solve!(alg::NMF.SPA{T}, X, W, H, _) where {T} = NMF.solve!(alg, X, W, H)

function custom_solve!(alg::NMF.MultUpdate{T}, X, W, H, simplex_H=false) where {T}
  if alg.obj == :mse
    nmf_skeleton!(NMF.MultUpdMSE(alg.update_H, alg.lambda_w, alg.lambda_h, sqrt(eps(T))), X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)
  else # alg.obj == :div
    nmf_skeleton!(NMF.MultUpdDiv(alg.update_H, alg.lambda_w, alg.lambda_h, sqrt(eps(T))), X, W, H, alg.maxiter, alg.verbose, alg.tol, simplex_H)
  end
end


# custom_solve!(alg, X, W, H, _) = NMF.solve!(alg, X, W, H)

"""
Modified internal high-level routine similar to that of NMF.jl
Added minimum volume constrained NMF as a usable algorithm
Added the ability to specify algorithm specific arguments
"""
function run_nmf(X::AbstractMatrix{T}, k::Integer;
  init::Symbol=:nndsvdar,
  initdata=nothing,
  alg::Symbol=:greedycd,
  maxiter::Integer=100,
  tol::Real=cbrt(eps(T) / 100),
  W0::Union{AbstractMatrix{T},Nothing}=nothing,
  H0::Union{AbstractMatrix{T},Nothing}=nothing,
  update_H::Bool=true,
  simplex_H::Bool=false,
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
  elseif alg == :mvc
    alginst = MinVolConstrained{T}(; maxiter, tol, verbose, update_H, kwargs...)
  elseif alg == :spa
    if init != :spa
      throw(ArgumentError("Invalid value for init, use :spa instead."))
    end
    alginst = NMF.SPA{T}(obj=:mse, kwargs...)
  else
    throw(ArgumentError("Invalid algorithm."))
  end
  return custom_solve!(alginst, X, W, H, simplex_H)
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
  simplex_W::Bool=false, performance_metric=result -> result.objvalue,
  kwargs...) where {T}

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
  _, min_idx = findmin(performance_metric, results)
  return results[min_idx]
end

