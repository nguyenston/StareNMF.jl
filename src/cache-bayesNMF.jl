using StanSample
using Distributions
using LinearAlgebra
using NMF
using DataFrames
using CSV
using JLD2

function main(K; overwrite=false)
  println("program start...")
  cancer_categories = Dict(
    # "skin" => "107-skin-melanoma-all-seed-1",
    # "ovary" => "113-ovary-adenoca-all-seed-1",
    "breast" => "214-breast-all-seed-1",
    # "liver" => "326-liver-hcc-all-seed-1",
    "lung" => "38-lung-adenoca-all-seed-1",
    # "stomach" => "75-stomach-adenoca-all-seed-1"
  )
  misspecification_type = Dict(
    "none" => "",
    "contaminated" => "-contamination-2",
    "overdispersed" => "-overdispersed-2.0",
    "perturbed" => "-perturbed-0.0025")
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

      data = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])$(misspecification_type[misspec]).tsv", DataFrame; delim='\t')
      X = Matrix{Int}(data[:, 2:end])
      println("cancer: $(cancer)\tmisspec: $(misspec)\tK: $(K)")
      hp = hyperpriors[:, "$(cancer)-$(misspec)"]
      stan_data = (; I=size(X, 1), J=size(X, 2), K, X, alpha=1.0, gamma0=2.0, gamma1=4.0, delta0=hp[1], delta1=hp[2])
      model = SampleModel("model-$(cancer)-$(misspec)-$(K)", stan_program)

      _ = stan_sample(model; data=stan_data, num_chains=16, num_samples=1000, num_warmups=4000, delta=0.98, max_depth=12, show_logging=true)
      result = read_samples(model, :dataframe)
      jldsave("../raw-cache-stan/synthetic/cache-stan-$(cancer_categories[cancer])$(misspecification_type[misspec])-$(K).jld2"; result)
    end
  end
end

main(parse(Int, ARGS[1]))
