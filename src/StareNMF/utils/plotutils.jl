using NMF
using DataFrames
using LinearAlgebra
using Hungarian
using Distributions
using Makie

"""
use makie to plot a mutation signature
"""
function signature_plot(gridpos, signature; title="")
  subfig = GridLayout()

  colors = [:blue, :black, :red, :grey, :green, :pink]
  barlabels = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
  color_index = collect(0:96-1) ./ 16 .|> floor .|> Int .|> x -> x + 1

  signature_axis = Axis(gridpos; xticksvisible=false, xticklabelsvisible=false, title=title,
    ytickformat=values -> ["$(round(v*100; digits=2))%" for v in values], limits=((-1, 97), (0, nothing)))
  label_axis = Axis(gridpos; xticks=(8:16:88, barlabels), xticksvisible=false,
    yticksvisible=false, yticklabelsvisible=false, xgridvisible=false, ygridvisible=false,
    bottomspinevisible=false, topspinevisible=false, leftspinevisible=false, rightspinevisible=false,
    xticklabelfont=:bold, limits=((-1, 97), (0, 1)))

  barplot!(signature_axis, 0.5:95.5, signature, color=colors[color_index], width=1)
  barplot!(label_axis, 8:16:88, fill(1, 6); width=16, gap=0.05, color=colors,
    bar_labels=barlabels, label_offset=4, label_font=:bold)

  subfig[1:2, 1] = [signature_axis, label_axis]
  rowsize!(subfig, 1, Aspect(1, 0.25))
  rowsize!(subfig, 2, Aspect(1, 0.25 / 30))
  rowgap!(subfig, 2)
  gridpos[] = subfig
  return subfig, signature_axis
end

"""
Returns a function that maps rho to the worst performing score
"""
function rho_performance_factory(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_results::Vector{NMF.Result{T}}, componentwise_losses;
  weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.2hdiff),
  w_metric=(w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)),
  h_metric=(h, h_gt) -> abs(h - h_gt) / h_gt) where {T<:Number}

  zero_points = componentwise_losses .|> l -> (length(l), maximum(l))
  sort!(zero_points; by=last)

  zero_points_filtered = [zero_points[1]]
  for zp in zero_points[2:end]
    if zp[1] < zero_points_filtered[end][1]
      push!(zero_points_filtered, zp)
    end
  end

  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  n_gt_sig = nrow(gt_loadings)

  GT_sig_loadings = sorted_loadings[:, 1]
  valid_results = filter(nmf_results) do r
    size(r.H)[1] <= n_gt_sig
  end

  worst_performing_inferrences = Dict()
  for r in valid_results
    W = r.W
    H = r.H
    K, N = size(H)

    # Matrix W as a dataframe with each column being a signature
    W_L1 = norm.(eachcol(W), 1)
    avg_inferred_loadings = dropdims(sum(Diagonal(W_L1) * H; dims=2); dims=2) / N

    w_diffs = Iterators.product(eachcol(W), eachcol(relevant_signatures)) .|> x -> w_metric(x...)
    h_diffs = Iterators.product(avg_inferred_loadings, GT_sig_loadings) .|> x -> h_metric(x...)
    combined_diffs = weighting_function.(w_diffs, h_diffs)
    assignment, _ = hungarian(combined_diffs)

    @assert !haskey(worst_performing_inferrences, K)
    worst_performing_inferrences[K] = maximum([combined_diffs[r, c] for (r, c) in enumerate(assignment)])
  end
  worst_performance_by_K = maximum(values(worst_performing_inferrences))
  return (rho) -> begin
    idx = findfirst(x -> x[2] >= rho, zero_points_filtered)
    K = isnothing(idx) ? zero_points_filtered[end][1] : zero_points_filtered[max(1, idx - 1)][1]
    return get(worst_performing_inferrences, K, worst_performance_by_K)
  end
end

"""
Makie plot of structurally aware loss with respect to rho and k
"""
function rho_k_losses(gridpos, componentwise_losses, rhos; lambda=0.01, plot_title="", rho_choice=Nothing)
  subfig = GridLayout()
  ax = Axis(gridpos; yscale=log10, xlabel=L"\rho", xlabelsize=20, ylabelsize=20,
    ylabel=L"\mathcal{R}^{\rho}", title=plot_title, limits=((0, maximum(rhos) * 1.0), nothing))
  for cwl in componentwise_losses
    K = length(cwl)
    stare_loss = [sum(max.(0, cwl .- rh)) + lambda * K for rh in rhos]
    lines!(ax, rhos, stare_loss, label="K=$(K)", cycle=[:color, :linestyle])
  end

  if rho_choice != Nothing
    stare_losses = [sum(max.(0, cwl .- rho_choice)) + lambda * length(cwl) for cwl in componentwise_losses]
    scatter!(ax, [rho_choice], [minimum(stare_losses)]; marker=:xcross, color=:black)
  end

  subfig[1, 1] = ax
  axislegend(ax; orientation=:vertical, valign=:top, nbanks=2)
  # subfig[1, 2] = Legend(gridpos, ax, "Legend")
  gridpos[] = subfig
  return subfig, ax
end

"""
A plot of that shows the minimum rho to get the structurally aware loss to be zero, w.r.t. the number of components K
"""
function rho_k_bottom(gridpos, componentwise_losses; plot_title="")
  zero_points = componentwise_losses .|> l -> (length(l), maximum(l))
  sort!(zero_points; by=last)

  zero_points_filtered = [zero_points[1]]
  for zp in zero_points[2:end]
    if zp[1] < zero_points_filtered[end][1]
      push!(zero_points_filtered, zp)
    end
  end
  points = map(zero_points_filtered) do (K, rho)
    Point2f(K, rho)
  end
  componentwise_div_axis = Axis(gridpos[1, 1];
    xticks=zero_points_filtered[end][1]:zero_points_filtered[1][1],
    title=plot_title, yticklabelcolor=:blue)
  plt1 = scatterlines!(componentwise_div_axis, points; color=:blue, marker=:utriangle, label="max componentwise divergence")

  points = map(zip(zero_points_filtered[1:end-1], zero_points_filtered[2:end])) do zps
    ((K1, rho1), (K2, rho2)) = zps
    Point2f(K1, rho2 - rho1)
  end
  ymin = minimum((x -> x[2]).(points))
  ymax = maximum((x -> x[2]).(points))
  gap = ymax / ymin + 0.1

  infer_quality_axis = Axis(gridpos[1, 1]; yscale=log10, yaxisposition=:right, yticklabelcolor=:orange, limits=(nothing, (ymin / gap^0.5, ymax * gap^0.5)))
  hidespines!(infer_quality_axis)
  hidexdecorations!(infer_quality_axis)
  plt2 = scatterlines!(infer_quality_axis, points; color=:orange, marker=:dtriangle, label="inferrence quality")

  axislegend(componentwise_div_axis, [plt1, plt2], ["max componentwise divergence", "inferrence quality differential"])
  linkxaxes!(componentwise_div_axis, infer_quality_axis)
end

"""
TODO: add documentation
TODO: finish implementing K > n_gt_sig cases
"""
function bubbles(gridpos, gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_results::Vector{NMF.Result{T}};
  weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.2hdiff),
  w_metric=(w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)),
  h_metric=(h, h_gt) -> abs(h - h_gt) / h_gt,
  colorrange=(0, 0.3),
  simplex_W=false) where {T<:Number}

  subfig = GridLayout()

  sorted_loadings = sort(gt_loadings, rev=true)
  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  n_gt_sig = nrow(gt_loadings)

  GT_sig = sorted_loadings[:, 2]
  GT_sig_loadings = sorted_loadings[:, 1]

  # if maxK is larger than the number of ground truth signatures, add padding to labels
  max_n_sig = max(n_gt_sig, maximum(r -> size(r.H, 1), nmf_results))
  ytick_padding = fill("", max_n_sig - n_gt_sig)

  ax = Axis(gridpos; limits=((0, max_n_sig + 1), (0, max_n_sig + 1)), yticks=(1:max_n_sig, [GT_sig; ytick_padding]),
    xticks=(0.5:length(nmf_results)+1, ["GT"; ["K = $(size(r.H)[1])" for r in nmf_results]]))

  conditioned_W_L1 = W -> simplex_W ? Diagonal(norm.(eachcol(W), 1)) : Diagonal(fill(1, size(W, 2)))
  max_inferred_loading = maximum([maximum(sum(conditioned_W_L1(r.W) * r.H; dims=2)) / size(r.H)[2] for r in nmf_results])
  max_radius = round(max(maximum(GT_sig_loadings), max_inferred_loading); sigdigits=2)

  strokewidth = 1
  colormap = :Blues_5
  highclip = :black
  radius = x -> 50 * sqrt(x / max_radius)
  points = Point2f.(0.5, 1:n_gt_sig)
  legendradiuses = [round(max_radius / 4; sigdigits=2), round(max_radius / 2; sigdigits=2), max_radius]
  markersizes = radius.(legendradiuses)
  group_size = [MarkerElement(; marker=:circle, color=:white, strokewidth, markersize=ms) for ms in markersizes]

  scatter!(ax, points; markersize=radius.(GT_sig_loadings), color=fill(0, n_gt_sig),
    colorrange, highclip, colormap, strokewidth)
  lines!(ax, [1, 1], [0, max_n_sig + 1]; linewidth=3)

  for (r_idx, r) in enumerate(nmf_results)
    W = r.W
    H = r.H
    K, N = size(H)

    # Matrix W as a dataframe with each column being a signature
    W_L1 = conditioned_W_L1(W)
    avg_inferred_loadings = dropdims(sum(W_L1 * H; dims=2); dims=2) / N

    w_diffs = Iterators.product(eachcol(W), eachcol(relevant_signatures)) .|> x -> w_metric(x...)
    h_diffs = Iterators.product(avg_inferred_loadings, GT_sig_loadings) .|> x -> h_metric(x...)
    combined_diffs = weighting_function.(w_diffs, h_diffs)
    assignment, _ = hungarian(combined_diffs)

    # assigning location for unmatched signatures
    unmatched_assignment = n_gt_sig + 1
    for i in eachindex(assignment)
      if assignment[i] == 0
        assignment[i] = unmatched_assignment
        unmatched_assignment += 1
      end
    end

    points = Point2f.(r_idx + 0.5, assignment)
    scatter!(ax, points; colorrange, colormap, strokewidth, highclip,
      markersize=radius.(avg_inferred_loadings), color=[assignment[i] <= n_gt_sig ? w_diffs[i, assignment[i]] : max(colorrange...) + 1.0 for i in 1:K])
  end
  legend = Legend(gridpos, group_size, string.(legendradiuses), "Mean Loading";
    tellheight=true, patchsize=(35, 35))
  colorbar = Colorbar(gridpos; colormap, colorrange, highclip,
    label="Cosine Error", alignmode=Outside(), halign=:left, size=40)

  subfig[1, 1] = ax
  subfig[1, 2][1, 1] = legend
  subfig[1, 2][2, 1] = colorbar
  gridpos[] = subfig
  return subfig, ax
end

"""
Score the nmf result on how good it's inferrences are. 
The score is computed by bipartite matching against ground truth 
  w.r.t. the cosine difference.
Returns max relative mean loading difference and maximum difference
"""
function compare_against_gt(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_result::NMF.Result{T};
  weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.2hdiff),
  w_metric=(w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)),
  h_metric=(h, h_gt) -> abs(h - h_gt) / h_gt) where {T<:Number}

  # loadings are sorted in order of decreasing importance
  sorted_loadings = sort(gt_loadings, rev=true)
  GT_sig_loadings = sorted_loadings[:, 1]

  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])
  W = nmf_result.W
  H = nmf_result.H
  _, N = size(H)

  W_L1 = norm.(eachcol(W), 1)
  avg_inferred_loadings = dropdims(sum(Diagonal(W_L1) * H; dims=2); dims=2) / N

  w_diffs = Iterators.product(eachcol(W), eachcol(relevant_signatures)) .|> x -> w_metric(x...)
  h_diffs = Iterators.product(avg_inferred_loadings, GT_sig_loadings) .|> x -> h_metric(x...)
  combined_diffs = weighting_function.(w_diffs, h_diffs)
  assignment, _ = hungarian(combined_diffs)

  return (maximum([h_diffs[i, gt] for (i, gt) in enumerate(assignment)]),
    maximum([w_diffs[i, gt] for (i, gt) in enumerate(assignment)]))
end

"""
Bipartite match inferred results against ground truth signatures
  w.r.t. the cosine difference. Plot the results side by side
"""
function signature_side2side(gt_loadings::DataFrame, gt_signatures::DataFrame, nmf_results::Vector{NMF.Result{T}};
  nmf_result_names::Vector{String}=fill("", length(nmf_results)),
  weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.2hdiff),
  w_metric=(w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)),
  h_metric=(h, h_gt) -> abs(h - h_gt) / h_gt,
  sigplot=signature_plot) where {T<:Number}

  # loadings are sorted in order of decreasing importance
  n_gt_sig = nrow(gt_loadings)
  sorted_loadings = sort(gt_loadings, rev=true)
  GT_sig_loadings = sorted_loadings[:, 1]

  relevant_signatures = Matrix(gt_signatures[:, sorted_loadings[:, 2]])

  fig = Figure(size=(600 * (length(nmf_results) + 1), 200 * n_gt_sig))
  for i in 1:n_gt_sig
    GT_sig = sorted_loadings[i, 2]
    GT_sig_loading = sorted_loadings[i, 1]
    sigplot(fig[i, 1], gt_signatures[:, GT_sig]; title="$(GT_sig), avg_loading=$(round(GT_sig_loading, digits=2))")
  end
  for (r_idx, r) in enumerate(nmf_results)
    W = r.W
    H = r.H
    _, N = size(H)

    # Matrix W as a dataframe with each column being a signature
    W_L1 = norm.(eachcol(W), 1)
    avg_inferred_loadings = dropdims(sum(H .* W_L1; dims=2); dims=2) / N
    W_dataframe = DataFrame(W ./ W_L1', ["$i" for i in 1:size(W)[2]])

    w_diffs = Iterators.product(eachcol(W), eachcol(relevant_signatures)) .|> x -> w_metric(x...)
    h_diffs = Iterators.product(avg_inferred_loadings, GT_sig_loadings) .|> x -> h_metric(x...)
    combined_diffs = weighting_function.(w_diffs, h_diffs)'
    assignment, _ = hungarian(combined_diffs)

    for (GT_sig_id, inferred_sig) in enumerate(assignment)
      if inferred_sig != 0
        diff = combined_diffs[GT_sig_id, inferred_sig]
        sigplot(fig[GT_sig_id, r_idx+1], W_dataframe[:, "$(inferred_sig)"];
          title="$(nmf_result_names[r_idx]) $(inferred_sig), diff=$(round(diff, digits=2)), avg_loading=$(round(avg_inferred_loadings[inferred_sig], digits=2))")
      end
    end
  end
  fig
end

"""
TODO: Add documentation
"""
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
    signature_plot(fig[inferred_sig, 1], W_dataframe[:, "$(inferred_sig)"];
      title="$(nmf_result_name) $(inferred_sig), diff=$(round(diff, digits=2)), avg_loading=$(round(avg_inferred_loadings[inferred_sig], digits=2))")
    signature_plot(fig[inferred_sig, 2], gt_signatures[:, GT_sig]; title="$(GT_sig), avg_loading=$(round(GT_sig_loading, digits=2))")
  end
  fig
end
