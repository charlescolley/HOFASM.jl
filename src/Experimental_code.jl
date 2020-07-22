

function build_list_of_approx_tensors_test(triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
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


    for (indices,og_angles) in triangles
        for (perm_indices,angles) in zip(permutations(indices),permutations(og_angles))

            aprox_angle1 = eps * (floor(angles[1] / eps) + .5)
            aprox_angle2 = eps * (floor(angles[2] / eps) + .5)
            aprox_angle3 = 180 - aprox_angle1 - aprox_angle2
            approx_angles = [aprox_angle1,aprox_angle2,aprox_angle3]

            indices = tuple(perm_indices[1],perm_indices[2],perm_indices[3])

            if avg_bins
                if haskey(bin_averages,approx_angles)
                    bin_averages[approx_angles] += [angles[1],angles[2],angles[3]]
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


function Make_HOFASM_tensor_pairs_test(H_indices::Array{Array{Int64,2},1},B_indices::Array{Array{Int64,2},1},
                            B_vals::Array{Array{Float64,1},1},n::Int,m::Int)
    marginalized_ten_pairs = Array{Tuple{SparseMatrixCSC{Float64,Int64},SparseMatrixCSC{Float64,Int64}},1}(undef,0)
    for (H_idx,B_idx,v) in zip(H_indices,B_indices,B_vals)
        push!(marginalized_ten_pairs,
              (perm_mode1_marginalization(H_idx,ones(size(H_idx,1)),[1,2,3],n),
               perm_mode1_marginalization(B_idx,v,[1,2,3],m))
             )
    end
    return marginalized_ten_pairs
end

function HOFASM_contraction!_test(tensor_pairs::Array{NTuple{2,SparseMatrixCSC{Float64,Int64}},1},
                             X_k::Array{Float64,2},X_k_1::Array{Float64,2}) #where {F <: AbstractFloat}

    n,m = size(X_k_1)

    for i=1:n
       for j=1:m
           X_k_1[i,j] = 0.0
       end
    end

    for (Hn,Bn) in tensor_pairs
         X_k_1 .+= (Bn*(Hn*X_k)' )'
    end

end
