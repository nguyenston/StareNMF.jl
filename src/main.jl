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

function cache_result_hyprunmix(; overwrite=false, nysamples=15, multiplier=1,
  dataset="urban", nmf_algs=["bssmf"], nmfargs=(), filenameappend="")
  println("program start...")
  cache_name = "nys=$(nysamples)-multiplier=$(multiplier)"

  println("start looping...")
  data = CSV.read("../hyperspectral-unmixing-datasets/$(dataset)/data.csv", DataFrame)
  X = Matrix{Int}(data)
  Base.Filesystem.mkpath("../result-cache-hyprunmix/$(dataset)/$(cache_name)/")
  for nmf_alg in nmf_algs
    if isfile("../result-cache-hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban.jld2") && !overwrite
      continue
    end

    ks = 1:9
    results = rank_determination(X / 1000, ks;
      nmfargs=(; alg=Symbol(nmf_alg), maxiter=50000, replicates=1, ncpu=1, nmfargs...))

    componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
    Threads.@threads for i in eachindex(results)
      r = results[i]
      componentwise_losses[i] = componentwise_loss(X, r.W * 1000, r.H; nysamples, approxargs=(; multiplier))
    end
    jldsave("../result-cache-hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-urban-$(filenameappend).jld2";
      results, componentwise_losses)
  end
end

const default_result_generation_synthetic = (;
  cache_name_prepend="",
  rgen=(cancer, misspec, X, ks, nmf_alg, nmfargs) -> begin
    results = rank_determination(X, ks;
      nmfargs=(; alg=Symbol(nmf_alg), replicates=16, ncpu=16, simplex_W=true, nmfargs...))
    return results
  end
)
const R_result_generation_synthetic = (;
  cache_name_prepend="musicatk-",
  rgen=(cancer, misspec, _, ks, nmf_alg, _) -> begin
    Ws = [CSV.read("../raw-cache-R/synthetic/$(nmf_alg)/$(cancer)-$(misspec)$(k)-W.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix
    Hs = [CSV.read("../raw-cache-R/synthetic/$(nmf_alg)/$(cancer)-$(misspec)$(k)-H.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix{Float64}
    results = NMF.Result{Float64}.(Ws, Hs, 0, true, 0)
    return results
  end
)
const stan_result_generation_synthetic = (;
  cache_name_prepend="stan-",
  rgen=(cancer, misspec, X, ks, _, _) -> begin
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
    D, N = size(X)

    chain_to_result = K -> begin
      chain = load("../raw-cache-stan/synthetic/cache-stan-$(cancer_categories[cancer])$(misspecification_type[misspec])-$(K).jld2")["result"]
      numsamples = nrow(chain)
      H = Iterators.product(1:K, 1:N) .|> ((k, n),) -> sum(chain[!, "theta.$(k).$(n)"]) / numsamples
      W = Iterators.product(1:D, 1:K) .|> ((d, k),) -> sum(chain[!, "r.$(k).$(d)"]) / numsamples
      return NMF.Result{Float64}(W, H, 0, true, 0)
    end

    results = chain_to_result.(ks)
    return results
  end
)
function cache_result_synthetic(; overwrite=false, nysamples=20, multiplier=200, result_generation=default_result_generation_synthetic, nmfargs=(), nmf_algs=[])
  println("program start...")
  cancer_categories = Dict(
    # "skin" => "107-skin-melanoma-all-seed-1",
    # "ovary" => "113-ovary-adenoca-all-seed-1",
    # "breast" => "214-breast-all-seed-1",
    # "liver" => "326-liver-hcc-all-seed-1",
    # "lung" => "38-lung-adenoca-all-seed-1",
    # "stomach" => "75-stomach-adenoca-all-seed-1"
    "breast-custom" => "450-breast-custom",
  )
  misspecification_type = Dict(
    "none" => "",
    # "contaminated" => "-contamination-2",
    # "overdispersed" => "-overdispersed-2.0",
    # "perturbed" => "-perturbed-0.0025"
  )
  cache_name_prepend, rgen = result_generation
  cache_name = "$(cache_name_prepend)nys=$(nysamples)-multiplier=$(multiplier)"
  nmf_algs = nmf_algs

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
      results = rgen(cancer, misspec, X, ks, nmf_alg, nmfargs)
      componentwise_losses = Vector{Vector{Float64}}(undef, length(results))
      Threads.@threads for i in eachindex(results)
        r = results[i]
        componentwise_losses[i] = componentwise_loss(X, r.W, r.H; nysamples, approxargs=(; multiplier))
      end
      jldsave("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; results, componentwise_losses)
    end
  end
end

const default_result_generation_real = (;
  cache_name_prepend="",
  rgen=(cancer, X, ks, nmf_alg, nmfargs) -> begin
    results = rank_determination(X, ks;
      nmfargs=(; alg=Symbol(nmf_alg), replicates=16, ncpu=16, maxiter=400000, simplex_W=true, nmfargs...))
    return results
  end
)
const R_result_generation_real = (;
  cache_name_prepend="musicatk-",
  rgen=(cancer, _, ks, nmf_alg, _) -> begin
    cancer_categories = Dict(
      "skin" => "Skin-Melanoma",
      "ovary" => "Ovary-AdenoCA",
      "breast" => "Breast",
      "liver" => "Liver-HCC",
      "lung" => "Lung-SCC",
      "stomach" => "Stomach-AdenoCA")
    Ws = [CSV.read("../raw-cache-R/real/$(nmf_alg)/$(cancer_categories[cancer])$(k)-W.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix
    Hs = [CSV.read("../raw-cache-R/real/$(nmf_alg)/$(cancer_categories[cancer])$(k)-H.csv", DataFrame)[:, 2:end] for k in ks] .|> Matrix{Float64}
    results = NMF.Result{Float64}.(Ws, Hs, 0, true, 0)
    return results
  end
)
const stan_result_generation_real = (;
  cache_name_prepend="stan-",
  rgen=(cancer, X, ks, _, _) -> begin
    cancer_categories = Dict(
      "skin" => "Skin-Melanoma",
      "ovary" => "Ovary-AdenoCA",
      "breast" => "Breast",
      "liver" => "Liver-HCC",
      "lung" => "Lung-SCC",
      "stomach" => "Stomach-AdenoCA")
    D, N = size(X)

    chain_to_result = K -> begin
      chain = load("../raw-cache-stan/real/cache-stan-$(cancer_categories[cancer])-$(K).jld2")["result"]
      numsamples = nrow(chain)
      H = Iterators.product(1:K, 1:N) .|> ((k, n),) -> sum(chain[!, "theta.$(k).$(n)"]) / numsamples
      W = Iterators.product(1:D, 1:K) .|> ((d, k),) -> sum(chain[!, "r.$(k).$(d)"]) / numsamples
      return NMF.Result{Float64}(W, H, 0, true, 0)
    end

    results = chain_to_result.(ks)
    return results
  end
)
function cache_result_real(; overwrite=false, nysamples=20, multiplier=200, result_generation=default_result_generation_real, ks=1:21, nmfargs=(), nmf_algs=[])
  println("program start...")
  cancer_categories = Dict(
    "skin" => "Skin-Melanoma",
    "ovary" => "Ovary-AdenoCA",
    "breast" => "Breast",
    "liver" => "Liver-HCC",
    "lung" => "Lung-SCC",
    "stomach" => "Stomach-AdenoCA")
  cache_name_prepend, rgen = result_generation
  cache_name = "$(cache_name_prepend)nys=$(nysamples)-multiplier=$(multiplier)"
  nmf_algs = nmf_algs

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

    results = rgen(cancer, X, ks, nmf_alg, nmfargs)

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
    # "skin" => "107-skin-melanoma-all-seed-1",
    # "ovary" => "113-ovary-adenoca-all-seed-1",
    # "breast" => "214-breast-all-seed-1",
    # "liver" => "326-liver-hcc-all-seed-1",
    # "lung" => "38-lung-adenoca-all-seed-1",
    # "stomach" => "75-stomach-adenoca-all-seed-1"
    "breast-custom" => "450-breast-custom",
  )
  misspecification_type = Dict(
    "none" => "",
    "contaminated" => "-contamination-2",
    "overdispersed" => "-overdispersed-2.0",
    "perturbed" => "-perturbed-0.0025")

  println("start looping...")
  for nmf_alg in nmf_algs
    fig = Figure(size=(1500, 1000))
    ax = Axis(fig[1, 1])
    rhos = collect(0:0.1:5)
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

        rho_performance = rho_performance_factory(loadings, signatures, results, componentwise_losses;
          weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.1hdiff))
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

function generate_plots_hyprunmix(; cache_name="nys=20-multiplier=1", dataset="urban",
  nmf_algs=["bssmf"], rhos=0:0.1:40, filenameappend="")
  println("program start...")
  signatures = CSV.read("../hyperspectral-unmixing-datasets/urban/signatures.csv", DataFrame)
  loadings = CSV.read("../hyperspectral-unmixing-datasets/urban/loadings.csv", DataFrame)
  nloadings = nrow(loadings)

  data = CSV.read("../hyperspectral-unmixing-datasets/$(dataset)/data.csv", DataFrame)
  X = Matrix{Int}(data) / 1000
  D, N = size(X)

  println("start looping...")
  w_metric = (w, w_gt) -> 1 - (normalize(w)' * normalize(w_gt)) |> (x -> isnan(x) ? 1.0 : x)
  for nmf_alg in nmf_algs
    Base.Filesystem.mkpath("../plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("../plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/svg")

    # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
    # data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
    # X = Matrix(data[:, 2:end])

    file = load("../result-cache-hyprunmix/$(dataset)/$(cache_name)/cache-$(nmf_alg)-hyprunmix-$(dataset)$(filenameappend).jld2")
    results = [r for r in file["results"]]
    println([r.converged for r in results])
    componentwise_losses = file["componentwise_losses"]

    valid_results = filter(results) do r
      size(r.H)[1] <= nloadings
    end
    fig = Figure(size=(1600, 2400))

    rho_k_losses(fig[2, 1], componentwise_losses, rhos)
    subfig_bubs, ax_bubs = bubbles(
      fig[1, 1],
      loadings,
      signatures,
      results;
      w_metric,
      weighting_function=(wdiff, hdiff) -> wdiff
    )
    bubs_legend_and_colorbar = GridLayout()
    bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


    k_labels = results .|> x -> size(x.H, 1)
    ax1 = Axis(fig; yscale=identity, xticks=(1.5:length(results)+0.5, ["K = $(i)" for i in k_labels]),
      yticks=0:length(results))
    ax2 = Axis(fig; yscale=identity, limits=(nothing, (0, nothing)),
      xticks=(1.5:length(results)+0.5, ["K = $(i)" for i in k_labels]))

    mrl_maxes = compare_against_gt.(
      [loadings],
      [signatures],
      valid_results;
      w_metric,
      weighting_function=(wdiff, hdiff) -> wdiff
    )

    modelargs = [norm(r.W * r.H - X) / sqrt(D * N) for r in results] .|> x -> (x,)
    model = (m, s) -> Normal(m, s)
    # modelargs = [() for _ in 1:6]
    # model = x -> Poisson(x)
    bic = [BIC(X, results[k]; model, modelargs=modelargs[k]) for k in eachindex(results)]
    bic_order = sortperm(bic) |> invperm
    lines!(ax1, 1.5:length(results)+0.5, bic_order; color=:red, label="BIC order")
    # lines!(ax1, 1.5:length(valid_results)+0.5, bic; label="BIC")

    lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; label="max difference", color=:orange)
    linkxaxes!(ax_bubs, ax1, ax2)
    subfig_bubs[1, 1] = ax1
    subfig_bubs[2, 1] = ax2
    subfig_bubs[3, 1] = ax_bubs


    subfig_bubs[1, 2] = Legend(fig, ax1)
    subfig_bubs[2, 2] = Legend(fig, ax2)
    subfig_bubs[3, 2] = bubs_legend_and_colorbar

    rowsize!(subfig_bubs, 3, Relative(1 // 2))
    rowsize!(fig.layout, 2, Relative(1 // 3))

    fig[0, :] = Label(fig, "Urban - Hyperspectral unmixing"; tellwidth=false, fontsize=30)

    save("../plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(dataset)$(filenameappend).pdf", fig)
    save("../plots/hyprunmix/$(dataset)/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(dataset)$(filenameappend).svg", fig)
  end
end

function generate_plots_synthetic(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv", "greedycd"], rho_choice=Nothing)
  println("program start...")
  signatures_unsorted = CSV.read("../synthetic-data-2023/alexandrov2015_signatures.tsv", DataFrame; delim='\t')
  signatures = sort(signatures_unsorted)
  cancer_categories = Dict(
    # "skin" => "107-skin-melanoma-all-seed-1",
    # "ovary" => "113-ovary-adenoca-all-seed-1",
    # "breast" => "214-breast-all-seed-1",
    # "liver" => "326-liver-hcc-all-seed-1",
    # "lung" => "38-lung-adenoca-all-seed-1",
    # "stomach" => "75-stomach-adenoca-all-seed-1"
    "breast-custom" => "450-breast-custom",
  )
  misspecification_type = Dict(
    "none" => "",
    # "contaminated" => "-contamination-2",
    # "overdispersed" => "-overdispersed-2.0",
    # "perturbed" => "-perturbed-0.0025"
  )

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)
    Base.Filesystem.mkpath("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("../plots/synthetic/$(cache_name)/composite-$(nmf_alg)/svg")

    for misspec in keys(misspecification_type)
      println("alg: $(nmf_alg)\tcancer: $(cancer)\tmisspec: $(misspec)")
      # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix(data[:, 2:end])

      file = load("../result-cache-synthetic/$(cache_name)/cache-$(nmf_alg)-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
      results = [r for r in file["results"]]
      componentwise_losses = file["componentwise_losses"]

      valid_results = filter(results) do r
        size(r.H)[1] <= nloadings
      end
      fig = Figure(size=(3200, 1600))
      rhos = 0:0.01:20

      rho_k_losses(fig[1, 2][1, 1], componentwise_losses, rhos; rho_choice)
      rho_k_bottom(fig[1, 2][2, 1], componentwise_losses)
      subfig_bubs, ax_bubs = bubbles(fig[1, 1], loadings, signatures, results;
        weighting_function=(wdiff, hdiff) -> wdiff + tanh(0.1hdiff), simplex_W=true)
      bubs_legend_and_colorbar = GridLayout()
      bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


      k_labels = results .|> x -> size(x.H, 1)
      ax1 = Axis(fig; yscale=log10, xticks=(1.5:length(results)+0.5, ["K = $(i)" for i in k_labels]),
        yaxisposition=:right, ytickcolor=:blue, yticklabelcolor=:blue)
      ax2 = Axis(fig; yscale=identity, limits=(nothing, (0, nothing)),
        xticks=(1.5:length(results)+0.5, ["K = $(i)" for i in k_labels]),
        ytickcolor=:orange, yticklabelcolor=:orange)
      ax3 = Axis(fig; yscale=identity, xticks=(1.5:length(results)+0.5, ["K = $(i)" for i in k_labels]),
        yticks=0:length(results))

      mrl_maxes = compare_against_gt.([loadings], [signatures], valid_results; weighting_function=(cd, ld) -> cd + tanh(0.1ld))
      plt1 = lines!(ax1, 1.5:length(valid_results)+0.5, [mm[1] for mm in mrl_maxes]; color=:blue)
      plt2 = lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; color=:orange)

      modelargs = ()
      model = x -> Poisson(x)
      bic = [BIC(X, results[k]; model, modelargs) for k in eachindex(results)]
      bic_order = sortperm(bic) |> invperm
      plt3 = lines!(ax3, 1.5:length(results)+0.5, bic_order; color=:red)

      linkxaxes!(ax_bubs, ax1, ax2, ax3)
      subfig_bubs[1, 1] = ax1
      subfig_bubs[1, 1] = ax2
      subfig_bubs[2, 1] = ax3
      subfig_bubs[3, 1] = ax_bubs


      subfig_bubs[1, 2] = Legend(fig, [plt1, plt2], ["max relative loading difference", "max cosine difference"])
      subfig_bubs[2, 2] = Legend(fig, [plt3], ["BIC Order"])
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
    # "skin" => "Skin-Melanoma",
    # "ovary" => "Ovary-AdenoCA",
    "breast" => "Breast",
    # "liver" => "Liver-HCC",
    "lung" => "Lung-SCC",
    # "stomach" => "Stomach-AdenoCA"
  )
  artifacts = "SBS" .* ["27", "43", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "95"]

  sig_names = filter(x -> !in(x, artifacts), names(signatures)[3:end])
  loadings = DataFrame(loadings=[1000 + i for i in 1:length(sig_names)], signatures=sig_names)
  nloadings = nrow(loadings)

  println("start looping...")
  for nmf_alg in nmf_algs, cancer in keys(cancer_categories)
    Base.Filesystem.mkpath("../plots/real/$(cache_name)/composite-$(nmf_alg)/pdf")
    Base.Filesystem.mkpath("../plots/real/$(cache_name)/composite-$(nmf_alg)/svg")

    println("alg: $(nmf_alg)\tcancer: $(cancer)")
    # jldsave("../result-cache/rho-k-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2"; rhos, ks, losses, results)
    data = CSV.read("../WGS_PCAWG.96.ready/$(cancer_categories[cancer]).tsv", DataFrame; delim='\t')
    X = Matrix(data[:, 2:end])

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
    subfig_bubs, ax_bubs = bubbles(fig[1, 1], loadings, signatures, valid_results; weighting_function=(cd, _) -> cd, simplex_W=true)
    bubs_legend_and_colorbar = GridLayout()
    bubs_legend_and_colorbar[1:2, 1] = subfig_bubs.content[2].content.content .|> x -> x.content


    ax2 = Axis(fig; yscale=identity, limits=(nothing, (0, nothing)),
      xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]))
    ax3 = Axis(fig; yscale=identity, xticks=(1.5:length(valid_results)+0.5, ["K = $(i)" for i in 1:length(valid_results)]),
      yticks=0:length(valid_results))

    mrl_maxes = compare_against_gt.([loadings], [signatures], valid_results; weighting_function=(cd, _) -> cd)
    lines!(ax2, 1.5:length(valid_results)+0.5, [mm[2] for mm in mrl_maxes]; label="max difference", color=:orange)

    modelargs = ()
    model = x -> Poisson(x)
    bic = [BIC(X, results[k]; model, modelargs) for k in 1:length(valid_results)]
    bic_order = sortperm(bic) |> invperm
    plt3 = lines!(ax3, 1.5:length(valid_results)+0.5, bic_order; color=:red)

    linkxaxes!(ax_bubs, ax2, ax3)
    subfig_bubs[1, 1] = ax2
    subfig_bubs[2, 1] = ax3
    subfig_bubs[3, 1] = ax_bubs


    subfig_bubs[1, 2] = Legend(fig, ax2)
    subfig_bubs[2, 2] = Legend(fig, [plt3], ["BIC Order"])
    subfig_bubs[3, 2] = bubs_legend_and_colorbar

    rowsize!(subfig_bubs, 3, Relative(6 // 8))

    fig[0, :] = Label(fig, "$(cancer_categories[cancer])", fontsize=30)

    save("../plots/real/$(cache_name)/composite-$(nmf_alg)/pdf/composite-$(nmf_alg)-$(cancer_categories[cancer]).pdf", fig)
    save("../plots/real/$(cache_name)/composite-$(nmf_alg)/svg/composite-$(nmf_alg)-$(cancer_categories[cancer]).svg", fig)
  end
end

# cache_result_synthetic(; overwrite=true, result_generation=stan_result_generation_synthetic,  nmf_algs=["stan"], nysamples=100, multiplier=150)
# cache_result_real(; nmf_algs=["bssmf"])
# cache_result_real(; result_generation=stan_result_generation_real, nmf_algs=["stan"])
# generate_plots_real(; cache_name="nys=20-multiplier=200", nmf_algs=["multdiv", "greedycd", "bssmf", "alspgrad"])
# generate_plots_real(; cache_name="musicatk-nys=20-multiplier=200", nmf_algs=["nmf", "lda", "nsnmf"])
# generate_plots_synthetic(; cache_name="stan-nys=100-multiplier=150", nmf_algs=["stan"], rho_choice=0.9)
# generate_rho_performance_plots_synthetic(; cache_name="stan-nys=100-multiplier=150", nmf_algs=["stan"])
# generate_plots_hyprunmix(; cache_name="nys=15-multiplier=1", nmf_algs=["greedycd"], rhos=0:0.1:80, filenameappend="-lambdaw=0.5-lambdah=1.5")
# cache_result_hyprunmix(; overwrite=true, nmf_algs=["greedycd"], 
#   nmfargs=(; init=:random, replicates=16, ncpu=16, maxiter=10000, lambda_w=0.0, lambda_h=0.11, simplex_H=true), 
#   filenameappend="lw=0.0-lh=0.11-simplexh")
cache_result_hyprunmix(; overwrite=true, nmf_algs=["alspgrad"],
  nmfargs=(; init=:random, replicates=16, ncpu=16, maxiter=10000, simplex_H=true), filenameappend="simplexh")
