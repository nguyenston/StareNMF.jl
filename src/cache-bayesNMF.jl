using StanSample
using Distributions
using LinearAlgebra
using NMF
using DataFrames
using CSV
using JLD2

function synthetic(K; overwrite=false)
  println("program start...")
  cancer_categories = Dict(
    # "skin" => "107-skin-melanoma-all-seed-1",
    # "ovary" => "113-ovary-adenoca-all-seed-1",
    # "breast" => "214-breast-all-seed-1",
    # "liver" => "326-liver-hcc-all-seed-1",
    # "lung" => "38-lung-adenoca-all-seed-1",
    # "stomach" => "75-stomach-adenoca-all-seed-1"
    "breast_custom" => "600-breast-custom-seed-1",
  )
  misspecification_type = Dict(
    "none" => "",
    # "contaminated" => "-contamination-2",
    # "overdispersed" => "-overdispersed-2.0",
    # "perturbed" => "-perturbed-0.0025"
  )
  stan_program = read("poisson-nmf.stan", String)
  hyperpriors = CSV.read("../synthetic-data-2023/bayes-nmf-hyperprior.tsv", DataFrame; delim="\t")
  Base.Filesystem.mkpath("../raw-cache-stan/synthetic/")
  println("start looping...")
  for cancer in keys(cancer_categories)
    loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)
    nloadings = nrow(loadings)

    for misspec in keys(misspecification_type)
      if isfile("../raw-cache-stan/synthetic/cache-stan-$(cancer_categories[cancer])$(misspecification_type[misspec])-$(K).jld2") && !overwrite
        continue
      end

      file = load("../result-cache-synthetic/nys=20-multiplier=200/cache-multdiv-$(cancer_categories[cancer])$(misspecification_type[misspec]).jld2")
      result = file["results"][K]

      Wraw = result.W .+ 0.1maximum(result.W)
      norm_coeff = eachcol(Wraw) .|> (x -> norm(x, 1))
      W = Wraw ./ norm_coeff'
      H = (result.H .+ 0.1maximum(result.H)) .* norm_coeff

      count_dataframe = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix{Int}(count_dataframe[:, 2:end])
      println("cancer: $(cancer)\tmisspec: $(misspec)\tK: $(K)")
      hp = hyperpriors[:, "$(cancer)-$(misspec)"]
      data = (; I=size(X, 1), J=size(X, 2), K, X, alpha=1.0, gamma0=2.0, gamma1=4.0, delta0=hp[1], delta1=hp[2])
      init = (; theta=H, r=collect(eachcol(W)), nu=rand(K), mu=rand(K))
      model = SampleModel("model-$(cancer)-$(misspec)-$(K)", stan_program)

      _ = stan_sample(model; data=data, init=init, num_chains=4, num_samples=1000, num_warmups=3000, delta=0.98, max_depth=12, show_logging=true)
      result = read_samples(model, :dataframe)
      jldsave("../raw-cache-stan/synthetic/cache-stan-$(cancer_categories[cancer])$(misspecification_type[misspec])-$(K).jld2"; result)
    end
  end
end

function real(K; overwrite=false)
  println("program start...")
  cancer_categories = Dict(
    # "skin" => "Skin-Melanoma",
    # "ovary" => "Ovary-AdenoCA",
    "breast" => "Breast",
    # "liver" => "Liver-HCC",
    # "lung" => "Lung-SCC",
    # "stomach" => "Stomach-AdenoCA"
  )
  stan_program = read("poisson-nmf.stan", String)
  hyperpriors = CSV.read("../WGS_PCAWG.96.ready/bayes-nmf-hyperprior.tsv", DataFrame; delim="\t")
  Base.Filesystem.mkpath("../raw-cache-stan/real/")
  println("start looping...")
  for cancer in keys(cancer_categories)
    if isfile("../raw-cache-stan/real/cache-stan-$(cancer_categories[cancer])-$(K).jld2") && !overwrite
      continue
    end

    file = load("../result-cache-real/nys=20-multiplier=200/cache-real-greedycd-$(cancer_categories[cancer]).jld2")
    result = file["results"][K]

    Wraw = result.W .+ 0.1maximum(result.W)
    norm_coeff = eachcol(Wraw) .|> (x -> norm(x, 1))
    W = Wraw ./ norm_coeff'
    H = (result.H .+ 0.1maximum(result.H)) .* norm_coeff

    count_dataframe = CSV.read("../WGS_PCAWG.96.ready/$(cancer_categories[cancer]).tsv", DataFrame; delim='\t')
    X = Matrix{Int}(count_dataframe[:, 2:end])
    println("cancer: $(cancer)\tK: $(K)")
    hp = hyperpriors[:, "$(cancer)"]
    data = (; I=size(X, 1), J=size(X, 2), K, X, alpha=1.0, gamma0=2.0, gamma1=4.0, delta0=hp[1], delta1=hp[2])
    init = (; theta=H, r=collect(eachcol(W)), nu=rand(K), mu=rand(K))
    model = SampleModel("model-$(cancer)-$(K)", stan_program)

    _ = stan_sample(model; data=data, init=init, num_chains=16, num_samples=1000, num_warmups=4000, delta=0.98, max_depth=12, show_logging=true)
    result = read_samples(model, :dataframe)
    jldsave("../raw-cache-stan/real/cache-stan-$(cancer_categories[cancer])-$(K).jld2"; result)
  end
end
synthetic(parse(Int, ARGS[1]))
