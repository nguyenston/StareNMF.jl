module Utils
export invcdf, plot_signature

using Distributions
using Makie

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
function plot_signature(gridpos, signatures, sig, title)
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
end
