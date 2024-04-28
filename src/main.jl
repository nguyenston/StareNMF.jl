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
  nmfargs = (; alg=:multdiv, nmfargs...) # default algorithm
  nmfargs = nmfargs.alg == :bssmf ? nmfargs : (; tol=1e-4, nmfargs...)
  for (i, k) in collect(enumerate(ks))
    result = threaded_nmf(Float64.(X), k; maxiter=200000, nmfargs...)
    results[i] = result
  end
  results
end

function cache_result_hyprunmix(; overwrite=false, nysamples=20, multiplier=200)
  println("program start...")
  cache_name = "nys=$(nysamples)-multiplier=$(multiplier)"
  nmf_algs = ["bssmf"]

  println("start looping...")
  data = CSV.read("../hyperspectral-unmixing-datasets/urban/data.csv", DataFrame)
  X = Matrix{Int}(data)
  Base.Filesystem.mkpath("../result-cache-hyprunmix/urban/$(cache_name)/")
  for nmf_alg in nmf_algs
    if isfile("../result-cache-hyprunmix/urban/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban.jld2") && !overwrite
      continue
    end

    ks = 1:9
    results = rank_determination(X, ks;
      nmfargs=(; alg=Symbol(nmf_alg), maxiter=5000, replicates=16, ncpu=16))

    componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
    Threads.@threads for i in eachindex(results)
      r = results[i]
      componentwise_losses[i] = componentwise_loss(X, r.W, r.H; nysamples, approxargs=(; multiplier))
    end
    jldsave("../result-cache-hyprunmix/urban/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban.jld2"; results, componentwise_losses)
  end
end

function cache_result_synthetic(; overwrite=false, nysamples=20, multiplier=200, r_mode=false)
  println("program start...")
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
  cache_name = r_mode ? "musicatk-nys=$(nysamples)-multiplier=$(multiplier)" : "nys=$(nysamples)-multiplier=$(multiplier)"
  nmf_algs = r_mode ? ["nmf"] : ["multdiv", "greedycd"]

  println("start looping...")
  Base.Filesystem.mkpath("../result-cache-synthetic/$(cache_name)/")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      if isfile("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2") && !overwrite
        continue
      end

      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      ks = 1:nloadings+3
      if r_mode
        Ws = [CSV.read("../raw-cache-R/synthetic/$(nmf_alg)/$(cancer)-$(misspec)$(k)-W.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix
        Hs = [CSV.read("../raw-cache-R/synthetic/$(nmf_alg)/$(cancer)-$(misspec)$(k)-H.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix{Float64}
        results = NMF.Result{Float64}.(Ws, Hs, 0, true, 0)
      else
        results = rank_determination(X, ks;
          nmfargs=(; alg=Symbol(nmf_alg), replicates=16, ncpu=16))
      end
      componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
      Threads.@threads for i in eachindex(results)
        r = results[i]
        componentwise_losses[i] = componentwise_loss(X, r.W, r.H; nysamples, approxargs=(; multiplier))
      end
      jldsave("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; results, componentwise_losses)
    end
  end
end

function cache_result_real(; overwrite=false, nysamples=20, multiplier=200, r_mode=false, ks=1:21)
  println("program start...")
  cancer_categories = Dict(
    "skin" => "Skin-Melanoma",
    "ovary" => "Ovary-AdenoCA",
    "breast" => "Breast",
    "liver" => "Liver-HCC",
    "lung" => "Lung-SCC",
    "stomach" => "Stomach-AdenoCA")
  cache_name = r_mode ? "musicatk-nys=$(nysamples)-multiplier=$(multiplier)" : "nys=$(nysamples)-multiplier=$(multiplier)"
  nmf_algs = r_mode ? ["nsnmf"] : ["multdiv", "greedycd"]

  println("start looping...")
  Base.Filesystem.mkpath("../result-cache-real/$(cache_name)/")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)

    println("alg: $(nmf_alg)\tcancer: $(cancer)\t")
    # skip if cache already exists
    if isfile("../result-cache-real/$(cache_name)/cache-real-$(nmf_alg)-$(cancer_categories[cancer]).jld2") && !overwrite
      continue
    end

    # compute NMF results
    data = CSV.read("../WGS_PCAWG.96.ready/$(cancer_categories[cancer]).tsv", DataFrame; delim='\t')
    X = Matrix(data[:, 2:end])

    if r_mode
      Ws = [CSV.read("../raw-cache-R/real/$(nmf_alg)/$(cancer_categories[cancer])$(k)-W.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix
      Hs = [CSV.read("../raw-cache-R/real/$(nmf_alg)/$(cancer_categories[cancer])$(k)-H.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix{Float64}
      results = NMF.Result{Float64}.(Ws, Hs, 0, true, 0)
    else
      results = rank_determination(X, ks;
        nmfargs=(; alg=Symbol(nmf_alg), replicates=16, ncpu=16, maxiter=400000))
    end

    # compute componentwise losses
    componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
    Threads.@threads for i in eachindex(results)
      r = results[i]
      componentwise_losses[i] = componentwise_loss(X, r.W, r.H; nysamples, approxargs=(; multiplier))
    end
    jldsave("../result-cache-real/$(cache_name)/cache-real-$(nmf_alg)-$(cancer_categories[cancer]).jld2"; results, componentwise_losses)
  end
end

function generate_rho_performance_plots_synthetic(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv", "greedycd"])
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
  for nmf_alg in nmf_algs
    fig = Figure(size=(1500, 1000))
    ax = Axis(fig[1, 1])
    rhos = collect(0:0.1:60)
    avg_performance = zeros(length(rhos))
    Base.Filesystem.mkpath("../plots/synthetic/$(cache_name)/rho-performances-$(nmf_alg)/")
    for cancer in keys(cancer_categories)
      local_fig = Figure(size=(1500, 1000))
      local_ax = Axis(local_fig[1, 1])
      local_avg_performance = zeros(length(rhos))


      loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
      nloadings = nrow(loadings)

      for misspec in keys(misspecification_type)
        println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
        # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
        # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
        # X = Matrix(data[:, 2:end])

        file = load("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
        results = [r for r in file["results"]]
        componentwise_losses = file["componentwise_losses"]

        rho_performance = rho_performance_factory(loadings, signatures, results, componentwise_losses; weighting_function=(cd, ld) -> cd + tanh(0.1ld))
        perf = rho_performance.(rhos)
        avg_performance .+= perf
        local_avg_performance .+= perf
        lines!(ax, rhos, rho_performance.(rhos); label="$(cancer)-$(misspec)", cycle=[:color, :linestyle], alpha=0.8)
        lines!(local_ax, rhos, rho_performance.(rhos); label="$(cancer)-$(misspec)", cycle=[:color, :linestyle], alpha=0.8)
      end
      lines!(local_ax, rhos, local_avg_performance / 4; label="Average", color=:black, linewidth=2)
      local_fig[1, 2] = Legend(local_fig, local_ax, "Legend")
      Label(local_fig[0, :], "Rho performances - $(cancer)", fontsize=25)
      save("../plots/synthetic/$(cache_name)/rho-performances-$(nmf_alg)/rho-performances-$(nmf_alg)-$(cancer).svg", local_fig)
    end
    lines!(ax, rhos, avg_performance / 24; label="Average", color=:black, linewidth=2)
    fig[1, 2] = Legend(fig, ax, "Legend")
    Label(fig[0, :], "Rho performances", fontsize=25)
    save("../plots/synthetic/$(cache_name)/rho-performances-$(nmf_alg)/rho-performances-$(nmf_alg).svg", fig)
    save("../plots/synthetic/$(cache_name)/rho-performances-$(nmf_alg)/rho-performances-$(nmf_alg).pdf", fig)
  end
end

function generate_plots_synthetic(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv", "greedycd"])
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
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)
    Base.Filesystem.mkpath("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/svg")

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
      # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      # X = Matrix(data[:, 2:end])

      file = load("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
      results = [r for r in file["results"]]
      componentwise_losses = file["componentwise_losses"]

      valid_results = filter(results) do r
        size(r.H)[1] <= nloadings
      end
      fig = Figure(size=(3200, 1600))
      rhos = 0:0.01:40

      rho_k_losses(fig[1, 2][1, 1], componentwise_losses, rhos)
      rho_k_bottom(fig[1, 2][2, 1], componentwise_losses)
      subfig_bubs, ax_bubs = bubbles(fig[1, 1], loadings, signatures, valid_results; weighting_function=(cd, ld) -> cd + tanh(0.1ld))
      bubs_legend_and_colorbar = GridLayout()
      bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


      ax1 = Axis(fig; yscale=log10, xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))
      ax2 = Axis(fig; yscale=identity, limits=(nothing, (0, nothing)),
        xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))

      mrl_maxes = compare_against_gt.([loadings], [signatures], valid_results; weighting_function=(cd, ld) -> cd + tanh(0.1ld))
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

      save("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).pdf", fig)
      save("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).svg", fig)
    end
  end
end

function generate_plots_real(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv", "greedycd"])
  println("program start...")
  signatures_unsorted = CSV.read("../WGS_PCAWG.96.ready/COSMIC_v3.4_SBS_GRCh38.tsv", DataFrame; delim='\t')
  signatures = sort(signatures_unsorted)
  cancer_categories = Dict(
    "skin" => "Skin-Melanoma",
    "ovary" => "Ovary-AdenoCA",
    "breast" => "Breast",
    "liver" => "Liver-HCC",
    "lung" => "Lung-SCC",
    "stomach" => "Stomach-AdenoCA")

  sig_names = names(signatures)[3:end]
  loadings = DataFrame(loadings=fill(2000, length(sig_names)), signatures=sig_names)
  nloadings = nrow(loadings)

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    Base.Filesystem.mkpath("../plots/real/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("../plots/real/$(cache_name)/composite-$(nmf_alg)/svg")

    println("alg: $(nmf_alg)\tcancer: $(cancer)")
    # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
    # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
    # X = Matrix(data[:, 2:end])

    file = load("../result-cache-real/$(cache_name)/cache-real-$(nmf_alg)-$(cancer_categories[cancer]).jld2")
    results = [r for r in file["results"]]
    println([r.converged for r in results])
    componentwise_losses = file["componentwise_losses"]

    valid_results = filter(results) do r
      size(r.H)[1] <= nloadings
    end
    fig = Figure(size=(3200, 3200))
    rhos = 0:0.01:40

    rho_k_losses(fig[1, 2][1, 1], componentwise_losses, rhos)
    rho_k_bottom(fig[1, 2][2, 1], componentwise_losses)
    subfig_bubs, ax_bubs = bubbles(fig[1, 1], loadings, signatures, valid_results; weighting_function=(cd, _) -> cd)
    bubs_legend_and_colorbar = GridLayout()
    bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


    ax2 = Axis(fig; yscale=identity, limits=(nothing, (0, nothing)),
      xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))

    mrl_maxes = compare_against_gt.([loadings], [signatures], valid_results; weighting_function=(cd, _) -> cd)
    lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; label="max difference", color=:orange)
    linkxaxes!(ax_bubs, ax2)
    subfig_bubs[1, 1] = ax2
    subfig_bubs[2, 1] = ax_bubs


    subfig_bubs[1, 2] = Legend(fig, ax2)
    subfig_bubs[2, 2] = bubs_legend_and_colorbar

    rowsize!(subfig_bubs, 2, Relative(7 // 8))

    fig[0, :] = Label(fig, "$(cancer_categories[cancer])", fontsize=30)

    save("../plots/real/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(cancer_categories[cancer]).pdf", fig)
    save("../plots/real/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(cancer_categories[cancer]).svg", fig)
  end
end

# cache_result_synthetic(; r_mode=true)
# generate_plots_real(; cache_name="musicatk-nys=20-multiplier=200", nmf_algs=["nsnmf"])
# generate_plots_synthetic(; cache_name="musicatk-nys=20-multiplier=200", nmf_algs=["nmf"])
# generate_rho_performance_plots_synthetic(; cache_name="musicatk-nys=20-multiplier=200", nmf_algs=["nmf"])
cache_result_hyprunmix()
