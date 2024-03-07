### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ e30fd15e-ba5d-4c7b-91a1-5238649b8b51
begin
	import Pkg
	# activate the shared project environment
	Pkg.activate(Base.current_project())
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
	push!(LOAD_PATH, "$(pwd())/StareNMF")
end

# ╔═╡ 983d24f4-c094-40ab-b96b-b81048990965
begin
	ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
	using PyCall
end

# ╔═╡ 5546058d-11be-48f0-a4c1-036dc9a80a3a
begin
	using StareNMF
	Utils = StareNMF.Utils
end

# ╔═╡ cffc8075-8134-4369-aad4-607f1f38970b
begin
	using DataFrames
	using CairoMakie
	using CSV
	using NPZ
end

# ╔═╡ 70b7302e-9598-477e-8709-72308857b2b2
using Makie: parent_scene, shift_project

# ╔═╡ 1e1b6c5f-241e-4d17-a24b-6745f4eed907
begin
	using NMF
	using KernelDensity
	using Distributions
	using Hungarian
end

# ╔═╡ 02db4599-b3d2-4913-bbc1-4324f8650b00
using LinearAlgebra

# ╔═╡ 627df66e-cbc7-11ee-0e3e-c770eabc3dde
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(80px, 5%);
    	padding-right: max(24px, 15%);
	}
</style>
"""

# ╔═╡ 6c6286bf-deaf-4e35-b98d-2df7cd701514
CairoMakie.activate!(type="svg")

# ╔═╡ c46c00c8-0e6c-4ea8-bb7a-6b54a29bd535
signatures_unsorted = CSV.read("../synthetic-data-2023/alexandrov2015_signatures.tsv", DataFrame; delim='\t')

# ╔═╡ c493008d-df0f-4869-bbf8-d0cfe215ad48
signatures = sort(CSV.read("../synthetic-data-2023/alexandrov2015_signatures.tsv", DataFrame; delim='\t'))

# ╔═╡ df32599c-d8d3-4166-ada4-b7e709a849fa
begin
	cancer_categories = Dict(
		"skin"    => "107-skin-melanoma-all-seed-1",
		"ovary"   => "113-ovary-adenoca-all-seed-1",
		"breast"  => "214-breast-all-seed-1",
		"liver"   => "326-liver-hcc-all-seed-1",
		"lung"    => "38-lung-adenoca-all-seed-1",
		"stomach" => "75-stomach-adenoca-all-seed-1")
	misspecification_type = Dict(
		"none"          => "",
		"contaminated"  => "-contamination-2",
		"overdispersed" => "-overdispersed-2.0",
		"perturbed"     => "-perturbed-0.0025")
	
end

# ╔═╡ ab330ce4-03ea-418a-87ff-dd1445104747
cancer = "liver"

# ╔═╡ b8bda4a6-639f-4dff-89b8-6473cec3fe3f
loadings = CSV.read("../synthetic-data-2023/synthetic-$(cancer_categories[cancer])-GT-loadings.csv", DataFrame; header=0)

# ╔═╡ fc462a6d-e59f-4287-a9ac-581cafb6f542
begin
	local fig = Figure(size=(1400, div(length(loadings[:, 2]) + 1, 2) * 210))
	for (i, sig) in enumerate(loadings[:, 2])
		(y, x) = divrem(i+1, 2)
		Utils.signature_plot(fig[y, x], signatures, sig; title=sig)
	end
	fig
end

# ╔═╡ bccd413c-5345-4df3-a24f-8c8be663035a
begin
	relevant_signatures = Matrix(signatures[:, loadings[:, 2]])
	svd(relevant_signatures).S
end

# ╔═╡ 9f02167e-1992-48e6-a2c0-fe48157fbd56
begin
	data = Dict()
	for (cancer, cancer_name) in pairs(cancer_categories)
		for (mis_type, mis_type_name) in pairs(misspecification_type)
			data[(cancer, mis_type)] =  CSV.read("../synthetic-data-2023/synthetic-$(cancer_name)$(mis_type_name).tsv", DataFrame; delim='\t')
		end
	end
end

# ╔═╡ eeebeb94-0d36-4043-8bec-868c9d19725c
begin
	X = Dict()
	for key in keys(data)
		X[key] = Matrix(data[key][:, 2:end])
	end
end

# ╔═╡ 172d00f7-42f3-419c-bc6a-e87d9691f691
all_loadings = npzread("../synthetic-data-2023/synthetic-113-ovary-adenoca-all-loadings.npy")

# ╔═╡ 93d11fb4-cd84-4eb7-bf98-de0839f51991
mutation_type = "overdispersed"

# ╔═╡ 620eb068-f906-4b3b-b374-55f50a587b6c
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

# ╔═╡ 627eb8be-d921-4059-81af-e5c59b4766e4
result_none = threaded_nmf(Float64.(X[(cancer, mutation_type)]), 8; alg=:greedycd, maxiter=200000, replicates=10, tol=1e-5, ncpu=8)

# ╔═╡ 4a2cdaa9-ab22-4cc2-9970-f75a1e78463f
typeof(result_none)

# ╔═╡ da97570d-42d2-4a54-82e4-454306e8b424
# ╠═╡ disabled = true
#=╠═╡
begin
	musical = pyimport("musical")
	local count_matrix = X[(cancer, mutation_type)]
	local lambda_tilde_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
	mvnmf_model = musical.DenovoSig(count_matrix, 
		min_n_components=8, # Minimum number of signatures to test
		max_n_components=8, # Maximum number of signatures to test
		init="random", # Initialization method
		method="mvnmf", # mvnmf or nmf
		n_replicates=10, # Number of mvnmf/nmf replicates to run per n_components
		ncpu=10, # Number of CPUs to use
		max_iter=100000, # Maximum number of iterations for each mvnmf/nmf run
		bootstrap=false, # Whether or not to bootstrap X for each run
		tol=1e-6, # Tolerance for claiming convergence of mvnmf/nmf
		verbose=0, # Verbosity of output
		normalize_X=false) # Whether or not to L1 normalize each sample in X before mvnmf/nmf
	mvnmf_model.fit()
end
  ╠═╡ =#

# ╔═╡ f6698f24-6d09-4501-94af-3c73652e2678
Axis(Figure())

# ╔═╡ b0416ef3-ffd8-4200-a66c-5a8c1e44ac09
#=╠═╡
mvnmf_model.W
  ╠═╡ =#

# ╔═╡ 6b87776f-11a7-441c-8b5b-7b23ac74ba87
begin
	local results = [result_none]
	local sig_types = ["Inferred Signature", "MuSiCal Signature"]
	# loadings are sorted in order of decreasing importance
	local sorted_loadings = sort(loadings, rev=true)
	local relevant_signatures = Matrix(signatures[:, sorted_loadings[:, 2]])
    local relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))
	local fig = Figure(size=(600 * (length(results) + 1), 200 * nrow(sorted_loadings)))

	for i in 1:nrow(sorted_loadings)
		GT_sig = sorted_loadings[i, 2]
		GT_sig_loading = sorted_loadings[i, 1]
		Utils.signature_plot(fig[i, 1], signatures, GT_sig; title="$(GT_sig), avg_loading=$(round(GT_sig_loading, digits=2))")
	end
	
	
	for (r_idx, r) in enumerate(results)
		W = r.W
		H = r.H
		K, N = size(H)
	
		W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))
		alignment_grid = 1 .- (relsig_normalized_L2' * W_normalized_L2)
		assignment, total_diff = hungarian(alignment_grid)

		# Matrix W as a dataframe with each column being a signature
		W_L1 = norm.(eachcol(W), 1)
		avg_inferred_loadings = sum(Diagonal(W_L1) * H; dims=2) / N
		W_dataframe = DataFrame(W * Diagonal(1 ./ W_L1), ["$i" for i in 1:size(W)[2]])
	
		println("avg diff, $(sig_types[r_idx]) = ", total_diff / K)
		for (GT_sig_id, inferred_sig) in enumerate(assignment)
			if inferred_sig != 0
				diff = alignment_grid[GT_sig_id, inferred_sig]
				Utils.signature_plot(fig[GT_sig_id, r_idx+1], W_dataframe, "$(inferred_sig)";
					title="$(sig_types[r_idx]) $(inferred_sig), diff=$(round(diff, digits=2)), avg_loading=$(round(avg_inferred_loadings[inferred_sig], digits=2))")
			end
		end
	end
	fig
end

# ╔═╡ e2ea8a1b-2d5c-4e4a-a0ad-c5c07d70a0a1
typeof(loadings)

# ╔═╡ ba274fa6-cf7f-47b4-a787-d7ed328871a3
# ╠═╡ disabled = true
#=╠═╡
begin
	local Kmax = 21
	local avg_diffs = fill(0., Kmax)
	local max_diffs = fill(0., Kmax)
	# loadings are sorted in order of decreasing importance
	local sorted_loadings = sort(loadings, rev=true)
	local relevant_signatures = Matrix(signatures[:, sorted_loadings[:, 2]])
	local relsig_normalized_L2 = relevant_signatures * Diagonal(1 ./ norm.(eachcol(relevant_signatures), 2))
	for k in 1:Kmax
		result_none = threaded_nmf(Float64.(X[(cancer, "overdispersed")]), k; alg=:greedycd, maxiter=200000, replicates=8, tol=1e-4)
		W = result_none.W
		W_normalized_L2 = W * Diagonal(1 ./ norm.(eachcol(W), 2))

		alignment_grid = 1 .- (relsig_normalized_L2' * W_normalized_L2)
		assignment, total_diff = hungarian(alignment_grid)
		
		max_dif = 0
		for (i, a) in enumerate(assignment)
			max_dif = a == 0 ? max_dif : max(max_dif, alignment_grid[i, a])
		end
		max_diffs[k] =  max_dif
		avg_diffs[k] = total_diff / k
	end
	local fig = Figure()
	local ax = Axis(fig[1, 1], xticks=1:Kmax)
	lines!(ax, 1:Kmax, avg_diffs, label="avg diff")
	lines!(ax, 1:Kmax, max_diffs, label="max diff")
	axislegend(ax)
	fig
end
  ╠═╡ =#

# ╔═╡ 978de3ab-cfe5-49b8-aea4-c1fff9e96c60
function rho_k_losses(losses, rho; krange=1:length(losses), plot_title="", kwargs...)
	fig = Figure(size=(800, 600))
	ax = Axis(fig[1, 1], yscale=log10, title=plot_title)
	for k in krange
		lines!(ax, rho, losses[k], label="k=$(k)")
	end
	Legend(fig[1, 2], ax, "Legend")
	fig
end

# ╔═╡ 6d35f0d2-4764-484f-bd3a-8773665ff780
function rank_determination(X, ks, rho; approx_type=StareNMF.KDEUniform, nmfargs=(), kwargs...)
	losses = Array{Vector{Float64}}(undef, length(ks))
	for (i, k) in collect(enumerate(ks))
		result = threaded_nmf(Float64.(X), k; alg=:multdiv, maxiter=200000, tol=1e-4, nmfargs...)
		losses[i] = StareNMF.structurally_aware_loss(X, result.W, result.H, rho; lambda=0.01, approx_type, kwargs...)
	end
	rho_k_losses(losses, rho; kwargs...)
end

# ╔═╡ c4f83f8a-2148-42e1-b5b2-b5582528b804
nmf_alg = :greedycd

# ╔═╡ 2a4ac0fe-8752-4cce-b4d6-4941a4e93599
# ╠═╡ disabled = true
#=╠═╡
rank_determination(X[(cancer, "none")], 1:nrow(loadings)+3, collect(0:0.01:10), 
	multiplier=500, plot_title="Not misspecified",
	nmfargs=(; alg=nmf_alg, replicates=5))
  ╠═╡ =#

# ╔═╡ a13b180c-b443-4e9d-8437-01d67d2c3691
# ╠═╡ disabled = true
#=╠═╡
rank_determination(X[(cancer, "contaminated")], 1:nrow(loadings)+3, collect(0:0.01:10), 
	multiplier=500, plot_title="Contaminated",
	nmfargs=(; alg=nmf_alg, replicates=5))
  ╠═╡ =#

# ╔═╡ 0d961f8c-5f4c-47e9-92b4-5193c051743b
# ╠═╡ disabled = true
#=╠═╡
rank_determination(X[(cancer, "perturbed")], 1:nrow(loadings)+3, collect(0:0.01:10), 
	multiplier=500, plot_title="Perturbed",
	nmfargs=(; alg=nmf_alg, replicates=5))
  ╠═╡ =#

# ╔═╡ 05e8316a-e797-4142-8211-5c8a5c11d4ca
# ╠═╡ disabled = true
#=╠═╡
rank_determination(X[(cancer, "overdispersed")], 1:nrow(loadings)+3, collect(0:0.01:10), 
	multiplier=500, plot_title="Overdispersed",
	nmfargs=(; alg=nmf_alg, replicates=5))
  ╠═╡ =#

# ╔═╡ 82eff34d-820d-4f08-aec5-73ba15aac2c9
# TODO: generate all plots, best match, buuble plots

# ╔═╡ Cell order:
# ╟─627df66e-cbc7-11ee-0e3e-c770eabc3dde
# ╠═e30fd15e-ba5d-4c7b-91a1-5238649b8b51
# ╠═983d24f4-c094-40ab-b96b-b81048990965
# ╠═5546058d-11be-48f0-a4c1-036dc9a80a3a
# ╠═cffc8075-8134-4369-aad4-607f1f38970b
# ╠═70b7302e-9598-477e-8709-72308857b2b2
# ╠═6c6286bf-deaf-4e35-b98d-2df7cd701514
# ╠═1e1b6c5f-241e-4d17-a24b-6745f4eed907
# ╠═02db4599-b3d2-4913-bbc1-4324f8650b00
# ╠═c46c00c8-0e6c-4ea8-bb7a-6b54a29bd535
# ╠═c493008d-df0f-4869-bbf8-d0cfe215ad48
# ╠═df32599c-d8d3-4166-ada4-b7e709a849fa
# ╠═ab330ce4-03ea-418a-87ff-dd1445104747
# ╠═b8bda4a6-639f-4dff-89b8-6473cec3fe3f
# ╠═fc462a6d-e59f-4287-a9ac-581cafb6f542
# ╠═bccd413c-5345-4df3-a24f-8c8be663035a
# ╠═9f02167e-1992-48e6-a2c0-fe48157fbd56
# ╠═eeebeb94-0d36-4043-8bec-868c9d19725c
# ╠═172d00f7-42f3-419c-bc6a-e87d9691f691
# ╠═93d11fb4-cd84-4eb7-bf98-de0839f51991
# ╠═620eb068-f906-4b3b-b374-55f50a587b6c
# ╠═627eb8be-d921-4059-81af-e5c59b4766e4
# ╠═4a2cdaa9-ab22-4cc2-9970-f75a1e78463f
# ╠═da97570d-42d2-4a54-82e4-454306e8b424
# ╠═f6698f24-6d09-4501-94af-3c73652e2678
# ╠═b0416ef3-ffd8-4200-a66c-5a8c1e44ac09
# ╠═6b87776f-11a7-441c-8b5b-7b23ac74ba87
# ╠═e2ea8a1b-2d5c-4e4a-a0ad-c5c07d70a0a1
# ╠═ba274fa6-cf7f-47b4-a787-d7ed328871a3
# ╠═978de3ab-cfe5-49b8-aea4-c1fff9e96c60
# ╠═6d35f0d2-4764-484f-bd3a-8773665ff780
# ╠═c4f83f8a-2148-42e1-b5b2-b5582528b804
# ╠═2a4ac0fe-8752-4cce-b4d6-4941a4e93599
# ╠═a13b180c-b443-4e9d-8437-01d67d2c3691
# ╠═0d961f8c-5f4c-47e9-92b4-5193c051743b
# ╠═05e8316a-e797-4142-8211-5c8a5c11d4ca
# ╠═82eff34d-820d-4f08-aec5-73ba15aac2c9
