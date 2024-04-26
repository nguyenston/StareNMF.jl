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
# ╠═╡ disabled = true
#=╠═╡
begin
	ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
	using PyCall
end
  ╠═╡ =#

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
	using JLD2
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

# ╔═╡ d5219149-5834-419c-a4e3-f6486658c70f
using MAT

# ╔═╡ 02db4599-b3d2-4913-bbc1-4324f8650b00
using LinearAlgebra

# ╔═╡ ac8a9354-0165-4cfb-8d87-4049eaccf680
using BSSMF

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

# ╔═╡ 492cf403-e689-4642-b5db-379d182b9d92
begin
	real_signatures = sort(CSV.read("../WGS_PCAWG.96.ready/COSMIC_v3.4_SBS_GRCh38.tsv", DataFrame; delim='\t'))
	local sig_names = names(real_signatures)[3:end]
	local loadings = DataFrame(loadings=fill(2000, length(sig_names)), signatures=sig_names)
	real_signatures
end

# ╔═╡ 9396d475-c2b7-4a88-b1d1-d6dd7ffc7981
length(names(real_signatures))

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

# ╔═╡ 9bdb281b-0161-4c55-84c4-bf4593b78b1c
Matrix(CSV.read("../WGS_PCAWG.96.ready/Breast.tsv", DataFrame; delim='\t')[:, 2:end])

# ╔═╡ efdf81f3-cc4c-4b9b-93d7-00825a7a1429
Symbol("multdiv")

# ╔═╡ 2d31c27f-8e9f-4b0c-8f16-1030245434cd
data[("skin", "none")][:, 1:end]

# ╔═╡ 172d00f7-42f3-419c-bc6a-e87d9691f691
all_loadings = npzread("../synthetic-data-2023/synthetic-113-ovary-adenoca-all-loadings.npy")

# ╔═╡ 93d11fb4-cd84-4eb7-bf98-de0839f51991
mutation_type = "overdispersed"

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
# ╠═╡ disabled = true
#=╠═╡
Axis(Figure())
  ╠═╡ =#

# ╔═╡ b0416ef3-ffd8-4200-a66c-5a8c1e44ac09
#=╠═╡
mvnmf_model.W
  ╠═╡ =#

# ╔═╡ c4f83f8a-2148-42e1-b5b2-b5582528b804
nmf_alg = :greedycd

# ╔═╡ 649162b9-f080-438d-ad98-4b175bd1afe4
begin
	local fig = Figure()
	local gl = GridLayout()
	local ax = Axis(fig)
	lines!(ax, 1:2, 1:2, label="loL")
	gl[1, 1][1, 1] = ax
	gl[1, 1][1, 2] = Axis(fig)
	gl[1, 2] = Legend(fig, ax)
	fig[1, 1] = gl
	display(fieldnames(typeof(gl)))
	display(fieldnames(typeof(gl[1, 1])))
	display(fieldnames(typeof(gl[1, 1].layout)))
	display(fieldnames(typeof(gl[1, 1].layout.content[1])))
	gl.content[1].content.content
end

# ╔═╡ 1ce40635-cde0-4c76-b932-12db32e5346a
end6 = matread("../hyperspectral-unmixing-datasets/urban/end6_groundTruth.mat")

# ╔═╡ 27f955b6-c5ca-4946-873a-1ab9b384e55b
sum(end6["M"]; dims=1)

# ╔═╡ a5c53190-4a3f-4402-898a-351a50de1a7a
CSV.write("../hyperspectral-unmixing-datasets/urban/signatures.csv", DataFrame(end6["M"], dropdims(end6["cood"]; dims=2)))

# ╔═╡ 7c9efdaa-0179-4d04-bd96-217eaa95a2d3
CSV.write("../hyperspectral-unmixing-datasets/urban/loadings.csv", DataFrame(loading=dropdims(sum(end6["A"]; dims=2); dims=2) / size(end6["A"], 2), signature=dropdims(end6["cood"]; dims=2)))

# ╔═╡ 81a64acd-01b6-49a0-8688-94bbb5c0d34a
size(end6["A"], 2)

# ╔═╡ 0d7d0447-f9dc-4313-b21d-48f5e635404e
a = rand(2, 3)

# ╔═╡ 5c804d0b-295c-4468-9f72-990091e1260a
b = rand(3, 4)

# ╔═╡ 19078865-0f66-4073-9e45-ee0e61e72fb5
collect(Iterators.product(eachcol(a), eachcol(b)))

# ╔═╡ f687170e-dbb0-4925-9026-d8ebe1b316c6


# ╔═╡ 7c1f1d4e-01e2-47a9-bba3-e585a2190bf7
DataFrame(loading=dropdims(sum(end6["A"]; dims=2); dims=2) / size(end6["A"], 2), signature=dropdims(end6["cood"]; dims=2))

# ╔═╡ 62b6385b-2848-42b4-aa30-889d25faf218
urban = matread("../hyperspectral-unmixing-datasets/urban/Urban_R162.mat")

# ╔═╡ 65f2a5d0-2902-473f-adcb-ddbf5e77b584
CSV.write("../hyperspectral-unmixing-datasets/urban/data.csv", Tables.table(urban["Y"]))

# ╔═╡ 05b059b2-35fd-44bf-b8cc-7e54ed87e0f0
Matrix(CSV.read("../hyperspectral-unmixing-datasets/urban/data.csv", DataFrame))

# ╔═╡ 1cb61efd-c351-446c-9245-9921c6b0cf20


# ╔═╡ Cell order:
# ╟─627df66e-cbc7-11ee-0e3e-c770eabc3dde
# ╠═e30fd15e-ba5d-4c7b-91a1-5238649b8b51
# ╠═983d24f4-c094-40ab-b96b-b81048990965
# ╠═5546058d-11be-48f0-a4c1-036dc9a80a3a
# ╠═cffc8075-8134-4369-aad4-607f1f38970b
# ╠═70b7302e-9598-477e-8709-72308857b2b2
# ╠═6c6286bf-deaf-4e35-b98d-2df7cd701514
# ╠═1e1b6c5f-241e-4d17-a24b-6745f4eed907
# ╠═d5219149-5834-419c-a4e3-f6486658c70f
# ╠═02db4599-b3d2-4913-bbc1-4324f8650b00
# ╠═c46c00c8-0e6c-4ea8-bb7a-6b54a29bd535
# ╠═c493008d-df0f-4869-bbf8-d0cfe215ad48
# ╠═492cf403-e689-4642-b5db-379d182b9d92
# ╠═9396d475-c2b7-4a88-b1d1-d6dd7ffc7981
# ╠═df32599c-d8d3-4166-ada4-b7e709a849fa
# ╠═ab330ce4-03ea-418a-87ff-dd1445104747
# ╠═b8bda4a6-639f-4dff-89b8-6473cec3fe3f
# ╠═fc462a6d-e59f-4287-a9ac-581cafb6f542
# ╠═bccd413c-5345-4df3-a24f-8c8be663035a
# ╠═9f02167e-1992-48e6-a2c0-fe48157fbd56
# ╠═eeebeb94-0d36-4043-8bec-868c9d19725c
# ╠═9bdb281b-0161-4c55-84c4-bf4593b78b1c
# ╠═efdf81f3-cc4c-4b9b-93d7-00825a7a1429
# ╠═2d31c27f-8e9f-4b0c-8f16-1030245434cd
# ╠═172d00f7-42f3-419c-bc6a-e87d9691f691
# ╠═93d11fb4-cd84-4eb7-bf98-de0839f51991
# ╠═da97570d-42d2-4a54-82e4-454306e8b424
# ╠═f6698f24-6d09-4501-94af-3c73652e2678
# ╠═b0416ef3-ffd8-4200-a66c-5a8c1e44ac09
# ╠═c4f83f8a-2148-42e1-b5b2-b5582528b804
# ╠═649162b9-f080-438d-ad98-4b175bd1afe4
# ╠═1ce40635-cde0-4c76-b932-12db32e5346a
# ╠═27f955b6-c5ca-4946-873a-1ab9b384e55b
# ╠═a5c53190-4a3f-4402-898a-351a50de1a7a
# ╠═7c9efdaa-0179-4d04-bd96-217eaa95a2d3
# ╠═81a64acd-01b6-49a0-8688-94bbb5c0d34a
# ╠═0d7d0447-f9dc-4313-b21d-48f5e635404e
# ╠═5c804d0b-295c-4468-9f72-990091e1260a
# ╠═19078865-0f66-4073-9e45-ee0e61e72fb5
# ╠═f687170e-dbb0-4925-9026-d8ebe1b316c6
# ╠═7c1f1d4e-01e2-47a9-bba3-e585a2190bf7
# ╠═62b6385b-2848-42b4-aa30-889d25faf218
# ╠═65f2a5d0-2902-473f-adcb-ddbf5e77b584
# ╠═05b059b2-35fd-44bf-b8cc-7e54ed87e0f0
# ╠═ac8a9354-0165-4cfb-8d87-4049eaccf680
# ╠═1cb61efd-c351-446c-9245-9921c6b0cf20
