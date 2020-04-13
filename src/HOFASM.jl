module HOFASM

using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Arpack


include("triangles.jl")
include("HOFASM_impl.jl")
include("HOM.jl")
include("experiments.jl")



end # module
