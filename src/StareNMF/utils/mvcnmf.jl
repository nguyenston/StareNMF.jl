# TODO: documentation
#
mutable struct MinVolConstrained{T}
  using NMF: CoordinateDescentState
  maxiter::Int
  verbose::Bool
  tol::T
  update_H::Bool

  maxsubiter::Int
  eta::T           # initial learning rate
  eta_decay::T     # rate of decay of learning rate, should be in range [0, 1]
  eta_tol::T       # a multiplier to eta, controls the rate of descent
  delta::T         # strength of column-stochasticity constraint
  lambda::T        # strength of volume regularization

  function MinVolConstrained{T}(;
    maxiter::Integer=100,
    verbose::Bool=false,
    tol::Real=cbrt(eps(T)),
    update_H::Bool=true,
    maxsubiter::Int=15,
    eta::T=convert(T, 1000),
    eta_decay::T=convert(T, 0.5),
    eta_tol::T=convert(T, 1e-4),
    delta::T=convert(T, 5),
    lambda::T=convert(T, 0.5)) where {T}

    return new{T}(maxiter, verbose, tol, update_H, maxsubiter, eta, eta_decay, eta_tol, delta, lambda)
  end
end

struct MinVolConstrainedUpd{T} <: NMF.NMFUpdater{T}
  eta::T
  eta_decay::T
  eta_tol::T
  delta::T
  lambda::T

  maxsubiter::Int
  update_H::Bool
end

mutable struct MinVolConstrainedState{T}
  # static quantities
  Xbar::Matrix{T},
  meanX::Matrix{T}
  meanXbar::Matrix{T}
  U::Matrix{T},    # PCA basis of X
  Ubar::Matrix{T}
  tau::T
  B::Matrix{T}
  C::Matrix{T}
  BUt::Matrix{T}
  BUbart::Matrix{T}

  function MinVolConstrainedState{T}(upd, X, W, H) where {T}
    D, N, K = NMF.nmf_checksize(X, W, H)
    Xbar = [X; upd.delta * ones(size(X, 2))]
    meanX = mean(X; dims=2)
    meanXbar = mean(Xbar; dims=2)

    U = fit(PCA, X; maxoutdim=K - 1) |> eigvecs
    Ubar = fit(PCA, transpose(Xbar); maxoutdim=K - 1) |> eigvecs

    tau = upd.lambda / factorial(big(K - 1))
    B = [zeros(1, K - 1); I]
    C = [ones(1, K); zeros(K - 1, K)]

    new{T}(X, meanX, meanXbar, Xbar, U, Ubar, tau, B, C, B * transpose(U), B * transpose(Ubar))
  end
end

prepare_state(upd::MinVolConstrainedUpd{T}, X, W, H) where {T} = MinVolConstrainedState{T}(upd, X, W, H)

function evaluate_objv(::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T}, X, W, H) where {T}
  norm(X - W * H) + s.tau / 2 * det(_Z(s, W))^2
end

function update_wh!(upd::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T},
  X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where {T}

  _update_W!(upd, s, X, W, H)
  if upd.update_H
    _update_H!(upd, s, s.Xbar, [W; upd.delta * ones(size(W, 2))], H)
  end
end

function _Z(state, W)
  if size(state.BUt, 2) == size(W, 1)
    return state.C + state.BUt * (W .- state.meanX)
  else
    @assert size(state.BUbart, 2) == size(W, 1) "W has invalid dimmension"
    return state.C + state.BUbart * (W .- state.meanXbar)
  end
end

function _update_W!(upd::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T},
  X, W, H, inner_iter=15) where {T}

  Ht = transpose(H)
  UBt = transpose(s.BUt)
  Z = _Z(s, W)
  Zt = transpose(Z)
  detZ = det(Z)
  old_obj = evaluate_objv(upd, s, X, W, H)


  W_new = Matrix{T}(undef, size(W))
  grad_frobenius = (W * H - X) * Ht
  for m in 0:inner_iter-1
    grad = grad_frobenius
    if !isapprox(detZ, 0; atol=1e-8, rtol=1e-5)
      grad = grad_frobenius + s.tau * detZ^2 * UBt * inv(Zt)          # adds volume regularization
    end

    W_new .= max.(0, W - upd.eta * upd.eta_decay^m * grad)
    new_obj = evaluate_objv(upd, s, X, W_new, H)
    armijo = upd.eta_tol * upd.eta * upd.eta_decay^m * sum(grad .* (W_new - W))

    if new_obj - old_obj <= armijo
      break
    end
  end

  W .= W_new
end

function _update_H!(upd::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T},
  X, W, H, inner_iter=15) where {T}

  Wt = transpose(W)
  old_obj = evaluate_objv(upd, s, X, W, H)

  H_new = Matrix{T}(undef, size(H))
  grad = Wt * (W * H - X)
  for m in 0:inner_iter-1
    H_new .= max.(0, H - upd.eta * upd.eta_decay^m * grad)
    new_obj = evaluate_objv(upd, s, X, W, H_new)
    armijo = upd.eta_tol * upd.eta * upd.eta_decay^m * sum(grad .* (H_new - H))

    if new_obj - old_obj <= armijo
      break
    end
  end

  H .= H_new
end
