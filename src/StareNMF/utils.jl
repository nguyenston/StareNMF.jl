module Utils
export invcdf, threaded_nmf, count_matrix_from_WH
export signature_plot, signature_side2side, signature_bestmatch, bubbles
export score_by_cosine_difference

using Distributions
using Makie
using NMF
using DataFrames
using LinearAlgebra
using Hungarian

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
function signature_plot(gridpos, signatures, sig; title="")
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

function bubbles(gridpos, gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_results::Vector{NMF.Result{T}}) where {T<:Number}
  subfig = GridLayout()

  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))
  n_gt_sig = nrow(gt_loadings)
  GT_sig = sorted_loadings[:, 2]
  GT_sig_loading = sorted_loadings[:, 1]
  valid_results = filter(nmf_results) do r
    size(r.H)[1] <= n_gt_sig
  end

  fig = Figure(size=(1600, 800))
  ax = Axis(gridpos; limits=((0, n_gt_sig + 1), (0, n_gt_sig + 1)), yticks=(1:n_gt_sig, GT_sig),
    xticks=(0.5:n_gt_sig+1, ["GT"; ["K = $(size(r.H)[1])" for r in valid_results]]))

  strokewidth = 1
  colormap = :dense
  colorrange = (0, 0.3)
  radius = x -> 50 * sqrt(x / maximum(GT_sig_loading))
  points = Point2f.(0.5, 1:n_gt_sig)
  legendradiuses = [1000, 2000, 4000]
  markersizes = radius.(legendradiuses)
  group_size = [MarkerElement(; marker=:circle, color=:white, strokewidth, markersize=ms) for ms in markersizes]

  scatter!(ax, points; markersize=radius.(GT_sig_loading), color=fill(0, n_gt_sig), colorrange, colormap, strokewidth)
  lines!(ax, [1, 1], [0, n_gt_sig + 1]; linewidth=3)

  for (r_idx, r) in enumerate(valid_results)
    W = r.W
    H = r.H
    K, N = size(H)

    W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))
    alignment_grid = 1 .- (W_normalized_L2' * relsig_normalized_L2)
    assignment, _ = hungarian(alignment_grid)

    # Matrix W as a dataframe with each column being a signature
    W_L1 = norm.(eachcol(W), 1)
    avg_inferred_loadings = dropdims(sum(Diagonal(W_L1) * H; dims=2); dims=2) / N
    points = Point2f.(r_idx + 0.5, assignment)
    scatter!(ax, points; markersize=radius.(avg_inferred_loadings), color=[alignment_grid[i, assignment[i]] for i in 1:K], colorrange, colormap, strokewidth)
  end

  subfig[1, 1] = ax
  subfig[1, 2][1, 1] = Legend(gridpos, group_size, string.(legendradiuses), "Mean Loading"; tellheight=true, patchsize=(35, 35))
  subfig[1, 2][2, 1] = Colorbar(gridpos; colormap, colorrange, label="Cosine Error", alignmode=Outside(), halign=:left, size=40)
  gridpos[] = subfig
end

"""
NMF.jl package's high level function nnmf, 
but can specify how many cpus to run in parallel for replicates
"""
function threaded_nmf(X, k; replicates=1, ncpu=1, kwargs...)
  results = Vector{NMF.Result{Float64}}(undef, replicates)
  c = Channel() do ch
    foreach(i -> put!(ch, i), 1:replicates)
  end

  Threads.foreach(c; ntasks=ncpu) do i
    results[i] = nnmf(X, k; kwargs..., replicates=1)
  end
  _, min_idx = findmin(x -> x.objvalue, results)
  return results[min_idx]
end

"""
Score the nmf result on how good it's inferrences are. 
The score is computed by bipartite matching against ground truth 
  w.r.t. the cosine difference.
Returns average difference and maximum difference
"""
function score_by_cosine_difference(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_result::NMF.Result{T}) where {T<:Number}
  # loadings are sorted in order of decreasing importance
  n_gt_sig = nrow(gt_loadings)
  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))
  W = nmf_result.W
  H = nmf_result.H
  K, _ = size(H)

  W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))
  alignment_grid = 1 .- (W_normalized_L2' * relsig_normalized_L2)
  assignment, total_score = hungarian(alignment_grid)
  return total_score / K, maximum([alignment_grid[i, gt] for (i, gt) in enumerate(assignment)])
end

"""
Bipartite match inferred results against ground truth signatures
  w.r.t. the cosine difference. Plot the results side by side
"""
function signature_side2side(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_results::Vector{NMF.Result{T}};
  nmf_result_names::Vector{String}=fill("", length(nmf_results))) where {T<:Number}

  # loadings are sorted in order of decreasing importance
  n_gt_sig = nrow(gt_loadings)
  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))

  fig = Figure(size=(600 * (length(nmf_results) + 1), 200 * n_gt_sig))
  for i in 1:n_gt_sig
    GT_sig = sorted_loadings[i, 2]
    GT_sig_loading = sorted_loadings[i, 1]
    Utils.signature_plot(fig[i, 1], gt_signatures, GT_sig; title="$(GT_sig), avg_loading=$(round(GT_sig_loading, digits=2))")
  end
  for (r_idx, r) in enumerate(nmf_results)
    W = r.W
    H = r.H
    _, N = size(H)

    W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))
    alignment_grid = 1 .- (relsig_normalized_L2' * W_normalized_L2)
    assignment, _ = hungarian(alignment_grid)

    # Matrix W as a dataframe with each column being a signature
    W_L1 = norm.(eachcol(W), 1)
    avg_inferred_loadings = sum(Diagonal(W_L1) * H; dims=2) / N
    W_dataframe = DataFrame(W * Diagonal(1 ./ W_L1), ["$i" for i in 1:size(W)[2]])

    for (GT_sig_id, inferred_sig) in enumerate(assignment)
      if inferred_sig != 0
        diff = alignment_grid[GT_sig_id, inferred_sig]
        Utils.signature_plot(fig[GT_sig_id, r_idx+1], W_dataframe, "$(inferred_sig)";
          title="$(nmf_result_names[r_idx]) $(inferred_sig), diff=$(round(diff, digits=2)), avg_loading=$(round(avg_inferred_loadings[inferred_sig], digits=2))")
      end
    end
  end
  fig
end

function signature_bestmatch(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_result::NMF.Result{T};
  nmf_result_name::String="") where {T<:Number}

  W = nmf_result.W
  H = nmf_result.H
  _, N = size(H)

  # Matrix W as a dataframe with each column being a signature
  W_L1 = norm.(eachcol(W), 1)
  avg_inferred_loadings = sum(Diagonal(W_L1) * H; dims=2) / N
  W_dataframe = DataFrame(W * Diagonal(1 ./ W_L1), ["$i" for i in 1:size(W)[2]])
  n_inferred_sig = ncol(W_dataframe)

  # loadings are sorted in order of decreasing importance
  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))

  fig = Figure(size=(600 * 2, 200 * n_inferred_sig))

  W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))
  alignment_grid = 1 .- (relsig_normalized_L2' * W_normalized_L2)
  assignment = map(x -> last(x), findmin.(eachcol(alignment_grid)))


  for (inferred_sig, GT_sig_id) in enumerate(assignment)
    diff = alignment_grid[GT_sig_id, inferred_sig]
    GT_sig = sorted_loadings[GT_sig_id, 2]
    GT_sig_loading = sorted_loadings[GT_sig_id, 1]
    Utils.signature_plot(fig[inferred_sig, 1], W_dataframe, "$(inferred_sig)";
      title="$(nmf_result_name) $(inferred_sig), diff=$(round(diff, digits=2)), avg_loading=$(round(avg_inferred_loadings[inferred_sig], digits=2))")
    Utils.signature_plot(fig[inferred_sig, 2], gt_signatures, GT_sig; title="$(GT_sig), avg_loading=$(round(GT_sig_loading, digits=2))")
  end
  fig
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
