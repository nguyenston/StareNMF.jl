include("StareNMF/StareNMF.jl")

using Distributions
using LinearAlgebra
using NMF
using DataFrames
using CSV
using JLD2
using .StareNMF
using .StareNMF.Utils

using CairoMakie
CairoMakie.activate!(type="svg")

function rho_k_losses(losses, rho; krange=1:length(losses), plot_title="", kwargs...)
  fig = Figure(size=(800, 600))
  ax = Axis(fig[1, 1], yscale=log10, title=plot_title)
  for k in krange
    lines!(ax, rho, losses[k], label="k=$(k)")
  end
  Legend(fig[1, 2], ax, "Legend")
  fig
end

function rank_determination(X, ks, rho; approx_type=StareNMF.KDEUniform, nmfargs=(), plotargs=(), kwargs...)
  losses = Array{Vector{Float64}}(undef, length(ks))
  results = Array{NMF.Result}(undef, length(ks))
  for (i, k) in collect(enumerate(ks))
    print("k=$(k)\t")
    result = threaded_nmf(Float64.(X), k; alg=:multdiv, maxiter=200000, tol=1e-4, nmfargs...)
    print("computing losses...\t\r")
    losses[i] = StareNMF.structurally_aware_loss(X, result.W, result.H, rho; lambda=0.01, approx_type, kwargs...)
    results[i] = result
  end
  rho_k_losses(losses, rho; plotargs..., krange=ks), losses, results
end

function main()
  signatures_unsorted = CSV.read("../synthetic-data-2023/alexandrov2015_signatures.tsv", DataFrame; delim='\t')
  signatures = sort(signatures_unsorted)
  cancer_categories = Dict(
    "skin" => "107-skin-melanoma-all-seed-1",
    "ovary" => "113-ovary-adenoca-all-seed-1",
    "breast" => "214-breast-all-seed-1",
    "liver" => "326-liver-hcc-all-seed-1",
    "lung" => "38-lung-adenoca-all-seed-1",
    "stomach" => "75-stomach-adenoca-all-seed-1")
  misspecification_type = Dict(
    "none" => "",
    "contaminated" => "-contamination-2",
    "overdispersed" => "-overdispersed-2.0",
    "perturbed" => "-perturbed-0.0025")

  println("start looping...")
  for cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)

    for misspec in keys(misspecification_type)
      println("cancer: $(cancer)\tmisspec: $(misspec)")
      if isfile("../plots/rho-k-plots/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
        continue
      end

      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])
      rhos = collect(0:0.01:10)
      ks = 1:nrow(loadings)+3

      fig, losses, results = rank_determination(X, ks, rhos; multiplier=500,
        plotargs=(; plot_title="$(cancer_categories[cancer])$(misspecification_type[misspec])"),
        nmfargs=(; alg=:greedycd, replicates=16, ncpu=16))
      jldsave("../plots/rho-k-plots/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
      save("../plots/rho-k-plots/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).svg", fig)
    end
  end
end

main()
