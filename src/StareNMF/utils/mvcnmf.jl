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
  eta_tol::T       # a multiplier to eta, controls the rate of descent, smaller => more strict
  delta::T         # strength of column-stochasticity constraint
  lambda::T        # strength of volume regularization

  function MinVolConstrained{T}(;
    maxiter::Integer=100,
    verbose::Bool=false,
    tol::Real=cbrt(eps(T)),
    update_H::Bool=true,
    maxsubiter::Int=15,
    eta::T=convert(T, 1e-3),
    eta_decay::T=convert(T, 0.5),
    eta_tol::T=convert(T, 1e-1),
    delta::T=convert(T, 5),
    lambda::T=convert(T, 1)) where {T}

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
  Xbar::Matrix{T}
  meanX::Matrix{T}
  meanXbar::Matrix{T}
  U::Matrix{T}    # PCA basis of X
  Ubar::Matrix{T}
  Ut::Matrix{T}
  tau::T

  function MinVolConstrainedState{T}(upd, X, W, H) where {T}
    _, _, K = NMF.nmf_checksize(X, W, H)
    Xbar = [X; upd.delta * ones(1, size(X, 2))]
    meanX = mean(X; dims=2)
    meanXbar = mean(Xbar; dims=2)

    U = fit(PCA, X; maxoutdim=K - 1, pratio=1.0) |> eigvecs
    Ubar = fit(PCA, Xbar; maxoutdim=K - 1, pratio=1.0) |> eigvecs

    tau = upd.lambda / factorial(big(K - 1))

    Ut = copy(transpose(U))

    new{T}(Xbar, meanX, meanXbar, U, Ubar, Ut, tau)
  end
end

prepare_state(upd::MinVolConstrainedUpd{T}, X, W, H) where {T} = MinVolConstrainedState{T}(upd, X, W, H)

function evaluate_objv(::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T}, X, W, H) where {T}
  norm(X - W * H)^2 / 2 + s.tau / 2 * det(_Z(W, s.Ut, s.meanX))^2
end

function update_wh!(upd::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T},
  X::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T}) where {T}

  _update_W!(upd, s, X, W, H, upd.maxsubiter)
  if upd.update_H
    Wbar = [W; upd.delta * ones(1, size(W, 2))]
    _update_H!(upd, s, s.Xbar, Wbar, H, upd.maxsubiter)
  end
end

function _Z(W, Ut, meanX)
  @assert size(Ut, 2) == size(W, 1) "W has invalid dimmension"
  W_tilde = Ut * (W .- meanX)
  return [ones(1, size(W, 2)); W_tilde]
end

function _update_W!(upd::MinVolConstrainedUpd{T}, s::MinVolConstrainedState{T},
  X, W, H, inner_iter=15) where {T}

  Ht = transpose(H)
  UBt = [zeros(size(s.U, 1)) s.U]
  Z = _Z(W, s.Ut, s.meanX)
  detZ = det(Z)
  old_obj = evaluate_objv(upd, s, X, W, H)

  W_new = Matrix{T}(undef, size(W))
  grad_frobenius = (W * H - X) * Ht
  grad = grad_frobenius
  if !isapprox(detZ, 0; atol=1e-8, rtol=1e-5)
    grad += s.tau * detZ^2 * UBt * transpose(inv(Z))        # adds volume regularization
  end
  for m in 0:inner_iter-1
    W_new .= max.(0, W - upd.eta * upd.eta_decay^m * grad)
    new_obj = evaluate_objv(upd, s, X, W_new, H)
    armijo = upd.eta_tol * upd.eta * upd.eta_decay^m * sum(grad .* (W_new - W))

    if new_obj - old_obj <= armijo
      break
    end
  end

  W .= W_new
end

function _update_H!(upd::MinVolConstrainedUpd{T}, ::MinVolConstrainedState{T},
  Xbar, Wbar, H, inner_iter=15) where {T}

  # no geometric penalty term here because it is independent from H and will cancel in the armijo check
  objv = (H) -> norm(Xbar - Wbar * H)^2 / 2
  Wbart = transpose(Wbar)
  old_obj = objv(H)

  H_new = Matrix{T}(undef, size(H))
  grad = Wbart * (Wbar * H - Xbar)
  for m in 0:inner_iter-1
    H_new .= max.(0, H - upd.eta * upd.eta_decay^m * grad)
    new_obj = objv(H_new)
    armijo = upd.eta_tol * upd.eta * upd.eta_decay^m * sum(grad .* (H_new - H))

    if new_obj - old_obj <= armijo
      break
    end
  end

  H .= H_new
end
