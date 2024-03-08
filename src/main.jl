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

function rho_k_losses(gridpos, losses, rhos; krange=1:length(losses), plot_title="")
  subfig = GridLayout()
  ax = Axis(gridpos, yscale=log10, title=plot_title)
  for k in krange
    lines!(ax, rhos, losses[k], label="k=$(k)")
  end

  subfig[1, 1] = ax
  subfig[1, 2] = Legend(gridpos, ax, "Legend")
  gridpos[] = subfig
end

function rank_determination(X, ks, rhos; approx_type=StareNMF.KDEUniform, nmfargs=(), plotargs=(), kwargs...)
  losses = Array{Vector{Float64}}(undef, length(ks))
  results = Array{NMF.Result}(undef, length(ks))
  for (i, k) in collect(enumerate(ks))
    result = threaded_nmf(Float64.(X), k; alg=:multdiv, maxiter=200000, tol=1e-4, nmfargs...)
    losses[i] = StareNMF.structurally_aware_loss(X, result.W, result.H, rhos; lambda=0.01, approx_type, kwargs...)
    results[i] = result
  end
  rho_k_losses(losses, rhos; plotargs..., krange=ks), losses, results
end

function main()
  println("program start...")
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
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      println("cancer: $(cancer)\tmisspec: $(misspec)")
      if isfile("../result-cache/cache-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
        continue
      end

      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      file = load("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
      results = file["results"]
      ks = file["ks"]
      rhos = file["rhos"]
      losses = file["losses"]

      componentwise_losses = Vector{Vector{Float64}}(undef, length(ks))
      Threads.@threads for i in eachindex(results)
        r = results[i]
        componentwise_losses[i] = componentwise_loss(X, r.W, r.H; multiplier=500)
      end
      jldsave("../result-cache/cache-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; results, componentwise_losses)


      # save("../plots/composite/composite-$(cancer_categories[cancer])$(misspecification_type[misspec]).svg", fig)
    end
  end
end

main()
