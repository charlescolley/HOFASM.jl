#=-----------------------------------------------------------------------------
   Contains the bulk of the subroutines For the HOFASM algorithm

   functions included are
   * synthetic_HOFASM
   * compute_angle_align_score
   * build_list_of_approx_tensor
   * build_angle_difference_tensors
   * build_index_and_bases_tensors
   * Make_HOFASM_tensor_pairs

-----------------------------------------------------------------------------=#



"""-----------------------------------------------------------------------------
    Runs a full synthetic correspondence problem alignment random points to
  perturbed points with std dev σ. Algorithm can be run with 3 different modes,
  which determine how the contractions are computed at each iteration of the
  graduated assignment routine, using the original specifications, or our sped
  up routines which utilize the kronecker mixed product property.

  Inputs:
  -------
  * n - (Int):
    The number of points to use in the correspondence problem
  * sigma - (Float64):
    The standard deviation of the pertubation applied to the source points
  * method - (optional String):
    a string indicating which method to use in the contraction phase of the
    algorithm.
    options:
        * - "new": Uses our new methods which use the mixed product property
                   to improve the runtime. This is the default option.
        * - "orig": Uses a list of explicitly built marginalized tensors
                    corresponding to each Hn ⊗ Bn pair
        * - "orig2": Uses the original list of tensor indices, and implicitly
                     marginalizes at each contraction phase.

  Outputs:
  --------
  * X - (Array{Float64,2}):
    The final assignment matrix from the graduated assignment iterations.
  * kron_time -(Float64):
    The time it takes to compute the pairs of marginalized tensors, returns 0.0
    when this step is unused.
  * iteration_time - (Float64):
    The run time of the graduated assignment step.
-----------------------------------------------------------------------------"""
function synthetic_HOFASM(n::Int,sigma::Float64,outliers::Int=0,scale::Float64=1.0;method="new",seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    source_points = randn(Float64,n+outliers,2)
    target_points = Array{Float64,2}(undef,n+outliers,2)
    for i::Int =1:n
        target_points[i,:] = source_points[i,:] + randn(2)*sigma
        target_points[i,:] *= scale
    end

    for i::Int=1:outliers
        target_points[n+i,:] = randn(2)
        source_points[n+i,:] = randn(2)
    end

    n += outliers
    m = n

    #shuffle the target points
    p = shuffle(1:n)
    target_points = target_points[p,:]


    #find all the triangles in between the points
    source_triangles = brute_force_triangles(source_points)
    target_triangles = brute_force_triangles(target_points)

    index_tensor_indices, index_tensor_vals , bases_tensor_indices, bases_tensor_vals =
        build_index_and_bases_tensors(source_triangles, target_triangles, 5.0)

#    return index_tensor_indices, index_tensor_vals , bases_tensor_indices, bases_tensor_vals
#    n = maximum([maximum(x) for x in index_tensor_indices if length(x) != 0])
#    m = maximum([maximum(x) for x in bases_tensor_indices if length(x) != 0])

    if method == "new"
        marg_ten_pairs, kron_time =
            @timed Make_HOFASM_tensor_pairs(index_tensor_indices,bases_tensor_indices,
                                            bases_tensor_vals,n,m)
        #=
        marg_ten_pairs, kron_time =
            @timed Make_HOFASM_tensor_pairs_test(index_tensor_indices,bases_tensor_indices,
                                            bases_tensor_vals,n,m)
        =#
        #return marg_ten_pairs
        x, iteration_time = @timed HOFASM_iterations(marg_ten_pairs,n,m)
    elseif method == "orig" #explicit marginalization
        marginalized_tensors,kron_time =
          @timed [perm_marginalize(Hn_ind,Bn_ind,Bn_val,n,m) for (Bn_ind,Bn_val,Hn_ind)
        in zip(bases_tensor_indices,bases_tensor_vals,index_tensor_indices)]

        x, iteration_time = @timed HOFASM_iterations(marginalized_tensors,n,m)

    elseif method =="orig2" # implicit marginalization
        kron_time = 0.0
        x, iteration_time = @timed HOFASM_iterations(bases_tensor_indices,
                                                     bases_tensor_vals,index_tensor_indices,n,m)
    elseif method == "SSoEfHOM"
        marg_ten_pairs, kron_time =
            @timed Make_HOFASM_tensor_pairs(index_tensor_indices,bases_tensor_indices,
                                            bases_tensor_vals,n,m)

        x, iteration_time = @timed SSoEfHOM(marg_ten_pairs,n,m)

    else
        throw(ArgumentError("valid method: must be 'new', 'orig', or 'orig2'"))
    end

    return reshape(x,n,m), kron_time, iteration_time, p

end


function compute_angle_align_score(angles1::Tuple{F,F,F}, angles2::Tuple{F,F,F}, sigma::F;testing_mode=false) where {F <: AbstractFloat}
    a1,a2,a3 = angles1
    ap1,ap2,ap3 = angles2
    #TODO: make this an option for special testing
    if testing_mode
        return ((a1 - ap1)^2+(a2 - ap2)^2+(a3 - ap3)^2) / sigma^2
    else
        return 4.5 - ((a1 - ap1)^2+(a2 - ap2)^2+(a3 - ap3)^2) / (6 * sigma^2)
    end
end


function build_list_of_approx_tensors(triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
    eps::F=1.0;avg_bins=false,test_mode=false) where {F <: AbstractFloat}
    """-------------------------------------------------------------------------
        Builds a dictionary linking approximate angles to that each triangle
      closest maps to and returns a list of pairs containing the indices and the
      approximated triangle angles for a given threshold eps.

    Inputs:
    -------
    * triangles - (List of Int 3-tuple, Float64 3-tuple, pairs ):
      Dictionary linking the indices of the hyperedges to the angles in the
      triangle.
    * eps - (optional float):
      The resolution for approximating angles (angle bin size).
      TODO: change var to bin_size
    * avg_bins - (optional bool):
      Indicates whether or not to alter the approximated angles to the average
      of each point within the bin, as opposed to the mid-point.

    Outputs:
    --------
    * Tensors - (Dict of 3-tuple -> list of 3-tuples):
      Dictionary linking approximate angles to the list of indices associated
      with the triangles approximated by each approximate angles.
    -------------------------------------------------------------------------"""
    #could probably improve performance using 3 tuples
    Tensors = Dict{Array{F,1},Array{Tuple{Int,Int,Int},1}}()

    if avg_bins
        new_approx_angle_tensors = Dict{Array{F,1},Array{Tuple{Int,Int,Int},1}}()
        bin_averages = Dict{Array{F,1},Array{F,1}}()
    end



    for (indices,angles) in triangles

        aprox_angle1 = eps * (floor(angles[1] / eps) + .5)
        aprox_angle2 = eps * (floor(angles[2] / eps) + .5)
        aprox_angle3 = 180 - aprox_angle1 - aprox_angle2
        approx_angles = [aprox_angle1,aprox_angle2,aprox_angle3]

        perm = sortperm(approx_angles)
        sort!(approx_angles)
        indices = tuple(indices[perm[1]],indices[perm[2]],indices[perm[3]])

        if avg_bins
            if haskey(bin_averages,approx_angles)
                bin_averages[approx_angles] += [angles[perm[1]],angles[perm[2]],angles[perm[3]]]
            else
                bin_averages[approx_angles] = [angles[perm[1]],angles[perm[2]],angles[perm[3]]]
            end
        end

        if haskey(Tensors,approx_angles)
            push!(Tensors[approx_angles],indices)
        else
            Tensors[approx_angles] = [indices]
        end
    end

    if avg_bins

        for (approx_angles,indices) in Tensors
            avg = bin_averages[approx_angles]/length(indices)
            new_approx_angle_tensors[avg] = indices
        end

        return new_approx_angle_tensors
    end

    return Tensors

end

# list of unique approximate angles and a list of indices matching to angles
function build_angle_difference_tensors(approx_angles::Array{F,2},
    triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
    sigma::F = 1.0;test_mode=false) where {F <: AbstractFloat}
    """-------------------------------------------------------------------------
        Computes a linking the approximate angles to a dictionary
    Inputs:
    -------
      * approx_angles - (2D array of Floats):
        Array of approximate angles to build all tensors against.  Each row is
        an has the approximate angles. Angles are unique up to permutations,
        indices may not be in sorted order.
      * triangles - (List of 3-tuple Int, 3-tuple Float pairs):
        pairs of indices and their corresponding angles. Assumes that the pairs
        are unique up to permutation.
      * sigma - (float)
        The angle tolerance threshold between whether or not to include the
        comparison between triangle, angles.
      * test_mode - (optional Bool)
    Outputs:
    --------
      * associated_tensors - (List of (3-tuple, Dict 3-tuple -> float) pairs):
        List of pairs of approximate angles paired with a dictionary linking
        indices of passed in triangles dictionary to the angle alignment score
        of between the angles and the approximated angle.
    -------------------------------------------------------------------------"""
    n, _ = size(approx_angles)
    associated_tensors = Dict{Tuple{F,F,F},Array{Tuple{Tuple{Int,Int,Int},F},1}}()
    tensor_index = 1

    for approx_angle in eachrow(approx_angles)

        compute_angle = tuple(approx_angle...)

        tensor_hyperedges = Array{Tuple{Tuple{Int,Int,Int},F},1}(undef,0)
        for (indices, image_angles) in triangles

            for (i,j,k) in permutations([1,2,3]) #permute image triangles
                index_perm = tuple(indices[i],indices[j],indices[k])
                angle_perm = tuple(image_angles[i],image_angles[j],image_angles[k])

                if test_mode
                    push!(tensor_hyperedges,
                          tuple(index_perm,compute_angle_align_score(angle_perm , compute_angle,
                                                                     sigma,
                                                                     testing_mode=test_mode)))
                else
                    if all([abs(a1 - a2) < 3 * sigma for (a1, a2) in zip(angle_perm, approx_angle)])
                        push!(tensor_hyperedges,
                              tuple(index_perm,compute_angle_align_score(angle_perm , compute_angle, sigma)))
                    end
                end
            end
        end

        associated_tensors[compute_angle] = tensor_hyperedges
    end

    return associated_tensors
end
"""----------------------------------------------------------------------------
        From the provided triangle dictionaries provided, produces the lists of
      hyper edges in the base tensors Bn, and the index tensors Hn. Index
      tensors are produced by finding the approximate center angles determined
      by the bin size from the triangles in image1. Base tensors are built by
      using the angle alignment score (equation 9 in HOFASM paper) between each
      approximate triangle and the triangles in image2. Produced tensors are
      remapped to new indices if any indices are untouched.

    Inputs:
    -------
    * image[1-2]_triangles - (list of (3-tuples,3-tuples) pairs):
      The set of triangles in each image, keys are the indices sorted in
      increasing order of the tensors linked to the corresponding angles in
      their triangle.
    * angle_bin_size - (Float):
      The bin size to approximate the angles with.
    * avg_bins -(optional bool):
      Indicates whether or not to replace the midpoint of each approximate
      angle with the average of the angles in each bin.
    * test_mode - (optional bool):
      Indicates whether to change the settings needed for a test mode which
      compares whether or not the HOFASM matrices can be equal to the HOM
      matrices.
    TODO: Update Outputs
    Outputs:
    --------
    * bases_tensors - (List of (3-tuple, Dict of 3-tuples -> float) pairs):
      List of approximate angles paired with base tensors computing the
      difference between the approximated triangle centers and the triangles
      in image2.
    * index_tensors - (List of (3-tuple, Dict of 3-tuples -> float) pairs):
      List of approximate angles paired with index tensors storing the indices
      of triangles in image1 associated with that approximated triangle.
 ----------------------------------------------------------------------------"""
function build_index_and_bases_tensors(image1_triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
    image2_triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
    angle_bin_size::F,avg_bins=false;test_mode=false) where {F <: AbstractFloat}

    #TODO: add avg_bins as arg for for testing
    if test_mode
        angle_bin_size=1e-16 #each angle gets their own bin
        avg_bins = true
    end

    approx_tensors =
        build_list_of_approx_tensors(image1_triangles, angle_bin_size,
                                     avg_bins = avg_bins,
                                     test_mode=test_mode)
    #=
    approx_tensors =
        build_list_of_approx_tensors_test(image1_triangles, angle_bin_size,
                                     avg_bins = avg_bins,
                                     test_mode=test_mode)
    =#
    approx_triangles =
      Matrix(reshape(collect(Iterators.flatten(keys(approx_tensors))),(3,length(approx_tensors)))')
    bases_tensors = build_angle_difference_tensors(approx_triangles,image2_triangles,
                                                   test_mode=test_mode)
    #build lists of indices and non-zeros values in order of sorted approximate angles

    #convert index tensor formats
    index_tensor_indices = Array{Array{Int,2},1}(undef,length(approx_tensors))
    index_tensor_vals = Array{Array{F,1},1}(undef,length(approx_tensors))
    bases_tensor_indices = Array{Array{Int,2},1}(undef,length(approx_tensors))
    bases_tensor_vals = Array{Array{F,1},1}(undef,length(approx_tensors))
    index = 1

    index_tensors_pairings::Array{Tuple{Array{Float64,1},Array{Tuple{Int64,Int64,Int64},1}},1} =
        sort([(k,v) for (k,v) in approx_tensors], by =x->x[1])
    bases_tensors_pairings::Array{Tuple{Tuple{Float64,Float64,Float64},Array{Tuple{Tuple{Int64,Int64,Int64},Float64},1}},1}=
        sort([(k,v) for (k,v) in bases_tensors],  by =x->x[1])

    for ((H_approx_angle,H_indices),(B_approx_angle,edges)) in zip(index_tensors_pairings,bases_tensors_pairings)

        @assert all([a1 == a2 for (a1,a2) in zip(H_approx_angle,B_approx_angle)])

        n = size(H_indices,1)
        m = length(edges)

        if n == 0 || m == 0
            #this produces a zero tensor
            continue
        end

        #preallocate needed memory
        H_matrix_indices = Array{Int,2}(undef,n,3)
        B_matrix_indices = Array{Int,2}(undef,m,3)
        edge_values = Array{F,1}(undef, m)


        for i =1:n
            i1,i2,i3 = H_indices[i]
            H_matrix_indices[i,1] = i1
            H_matrix_indices[i,2] = i2
            H_matrix_indices[i,3] = i3
        end

        for i =1:m
            (i1,i2,i3),val = edges[i]
            B_matrix_indices[i,1] = i1
            B_matrix_indices[i,2] = i2
            B_matrix_indices[i,3] = i3
            edge_values[i] = val
        end


        index_tensor_indices[index] = H_matrix_indices
        #TODO: may not need
        index_tensor_vals[index] = ones(n)
        bases_tensor_indices[index] = B_matrix_indices
        bases_tensor_vals[index] = edge_values
        index += 1

    end

    return index_tensor_indices[1:(index-1)], index_tensor_vals[1:(index-1)] , bases_tensor_indices[1:(index-1)], bases_tensor_vals[1:(index-1)]
end


function Make_HOFASM_tensor_pairs(H_indices::Array{Array{Int64,2},1},B_indices::Array{Array{Int64,2},1},
                            B_vals::Array{Array{Float64,1},1},n::Int,m::Int)
    marginalized_ten_pairs = Array{Tuple{SparseMatrixCSC{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}(undef,0)
    for (H_idx,B_idx,v) in zip(H_indices,B_indices,B_vals)
        for p in permutations((1,2,3))
            push!(marginalized_ten_pairs,
                  (perm_mode1_marginalization(H_idx,ones(size(H_idx,1)),p,n),
                   perm_mode1_marginalization(B_idx,v,p,m))
                 )
        end
    end
    return marginalized_ten_pairs
end
