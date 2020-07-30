#module HOFASM

using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Arpack
using Distributed

using ImageFeatures  #used for demos



include("triangles.jl")
include("contraction.jl")
include("GraduatedAssignment.jl")
include("Experimental_code.jl")
include("HOFASM_impl.jl")
include("HOM.jl")
include("experiments.jl")
#include("test/test_code.jl")



#end # module
