module HOFASM

using LinearAlgebra
using Combinatorics
using SparseArrays
using Statistics
using Random
using Arpack
using Distributed
using ImageFeatures  #used for demos



#
# Primary Experiment Drivers
#

export distributed_timing_experiments
export distributed_accuracy_experiments
export timing_experiments
export accuracy_experiments
export synthetic_HOM
export synthetic_HOFASM
export build_assignment

#
# Routines to help using our code in your own experiments
#

export align_photos
export align_embeddings



include("triangles.jl")
include("contraction.jl")
include("GraduatedAssignment.jl")
include("Experimental_code.jl")
include("HOFASM_impl.jl")
include("HOM_impl.jl")
include("experiments.jl")
#include("test/test_code.jl")



end # module
