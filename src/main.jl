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

function rank_determination(X, ks; nmfargs=())
  results = Array{NMF.Result}(undef, length(ks))
  for (i, k) in collect(enumerate(ks))
    result = threaded_nmf(Float64.(X), k; alg=:multdiv, maxiter=200000, tol=1e-4, nmfargs...)
    results[i] = result
  end
  results
end

function cache_result(overwrite=false)
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
  nmf_algs = ["multdiv", "greedycd"]

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      if isfile("../result-cache/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2") && !overwrite
        continue
      end

      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      results = rank_determination(X, 1:nloadings+3;
        nmfargs=(; alg=:multdiv, replicates=16, ncpu=16))
      componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
      Threads.@threads for i in eachindex(results)
        r = results[i]
        componentwise_losses[i] = componentwise_loss(X, r.W, r.H; approxargs=(; multiplier=500))
      end
      jldsave("../result-cache/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; results, componentwise_losses)
    end
  end
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
  nmf_algs = ["multdiv", "greedycd"]

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
      # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      # X = Matrix(data[:, 2:end])

      file = load("../result-cache/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
      results = [r for r in file["results"]]
      componentwise_losses = file["componentwise_losses"]

      valid_results = filter(results) do r
        size(r.H)[1] <= nloadings
      end
      fig = Figure(size=(3200, 1600))
      rhos = 0:0.01:40

      rho_k_losses(fig[1, 2][1, 1], componentwise_losses, rhos)
      rho_k_bottom(fig[1, 2][2, 1], componentwise_losses)
      subfig_bubs, ax_bubs = bubbles(fig[1, 1], loadings, signatures, valid_results)
      bubs_legend_and_colorbar = GridLayout()
      bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


      ax1 = Axis(fig; yscale=log10, xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))
      ax2 = Axis(fig; yscale=identity, xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))

      mrl_maxes = score_by_cosine_difference.([loadings], [signatures], valid_results)
      lines!(ax1, 1.5:length(valid_results)+0.5, [mm[1] for mm in mrl_maxes]; label="max relative loading difference")
      lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; label="max difference", color=:orange)
      linkxaxes!(ax_bubs, ax1, ax2)
      subfig_bubs[1, 1] = ax1
      subfig_bubs[2, 1] = ax2
      subfig_bubs[3, 1] = ax_bubs


      subfig_bubs[1, 2] = Legend(fig, ax1)
      subfig_bubs[2, 2] = Legend(fig, ax2)
      subfig_bubs[3, 2] = bubs_legend_and_colorbar

      rowsize!(subfig_bubs, 3, Relative(1 // 2))

      fig[0, :] = Label(fig, "$(cancer_categories[cancer])$(misspecification_type[misspec])", fontsize=30)

      save("../plots/composite-$(nmf_alg)/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).pdf", fig)
      save("../plots/composite-$(nmf_alg)/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).svg", fig)
    end
  end
end

main()
