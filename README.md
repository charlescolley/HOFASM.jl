HOFASM.jl
=====

julia (v1.3) implementations for the [HOFASM](https://www.ri.cmu.edu/pub_files/2014/3/Manuscript_Soonyong.pdf) and [HOM](https://ieeexplore.ieee.org/document/5432196) point correspondence algorithms. our HOFASM implementation makes use of the mixed contraction property of the tensor Kronecker product in the marginalization scheme to produce a 10x runtime improvement. 


Contents
========
 * HOFASM.jl: 
   Top file
 * HOFASM_impl.jl: 
   Routines for computing the base and index tensors from source and target triangles. Contains routines to set up synthetic experiments used in testing. 
 * HOM.jl: 
   Implementation files for the HOM algorithm, primarily used for testing and small baseline performance tests.
   **WARNING:** Code explicitly builds loss tensors in O(n^6) time and memory. 
 * contraction.jl:     
   Routines for computing the tensor contractions in the gradients
 * experiments.jl:
 Routines for running alignments between images, or the synthetic experiment framework developed by [Zass and Shashua](https://www.cse.huji.ac.il/~shashua/papers/matching-cvpr08.pdf). Currently supports testing for outliers, scalings, and normal noise. Note that Distributed routines must be run in the src folder because of how the @everywhere macro is being used. 
* triangles.jl: 
  Contains functions for computing triangle angles between the points. Planning to add in methods to build triangles from Delaunay mesh and nearest neighbor graphs. 
    
Dependencies
===========
[LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), [SparseArrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/), and [Arpack](https://github.com/JuliaLinearAlgebra/Arpack.jl) for any sparse numerical linear algebra routines needed, coupled with [Combinatorics](https://github.com/JuliaMath/Combinatorics.jl) for implicit transpose operations. 

[Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/) and [Random](https://docs.julialang.org/en/v1/stdlib/Random/) for synthetic experiments. [Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/) is used for the distributed experiment drivers. 

[ImageFeatures](https://juliaimages.org/ImageFeatures.jl/stable/) to find features in test images produced by users. 


