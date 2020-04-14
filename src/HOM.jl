
function synthetic_HOM(n::Int,sigma::Float64)

    d = 2
    source_points = randn(Float64,n,d)
    target_points = Array{Float64,2}(undef,n,d)
    for i =1:n
        target_points[i,:] = source_points[i,:] + randn(d)*sigma
    end

    #shuffle the target points
    p = shuffle(1:n)
    target_points = target_points[p,:]

    #find all the triangles in between the points
    source_triangles = brute_force_triangles(source_points)
    target_triangles = brute_force_triangles(target_points)

#    return source_triangles, target_triangles

    indices, vals = produce_HOM_tensor(source_triangles,target_triangles,n,tol=1e-12)

#    return indices, vals

    marginalized_tensor = sym_mode1_marginalization(indices,vals,n^2)
    #marg_result = reshape(HOM_graduated_assignment(marginalized_tensor),n,n)
    #lam, vec, _ = eigs(marginalized_tensor,nev=1)

    Graduated_Assignment_res, runtime = @timed Array(reshape(HOM_graduated_assignment(marginalized_tensor),n,n)')

    return Graduated_Assignment_res, runtime , p
    #ten_result = reshape(HOM_graduated_assignment(ssten.COOTen(indices,vals)),n,n)
#    return marginalized_tensor,reshape(vec,n,n), marg_result, ten_result, p

end

"""-------------------------------------------------------------------------
     produces the angular difference between all pairs of triangles that
   meet the sparsification tolerance. Final output tensor is a symmetric
   tensor which computes the elements with

     C_{ii',jj',kk'} =
          exp((|θ_i - θ_i'|^2 + |θ_j - θ_j'|^2 + |θ_k - θ_k'|^2)/ϵ^2)

    Final output is a symmetric COO formatted tensor.

    Inputs:
    -------
    * image[1-2]_triangles - (list of (3-tuples,3-tuples) pairs):
      The set of triangles in each image, keys are the indices sorted in
      increasing order of the tensors linked to the corresponding angles in
      their triangle.
    * m - (Int):
      The number of points in image2, needed to format the final indices.
    * tol - (Float64)
      The sparsification tolerance as to whether or not to include that pair
      of triangles in the final tensor.

    Output:
    -------
     * HOM_indices - (Array{Int,2}):
       All the indices of the output tensor up to permutations (entries in
       sorted order).
     * HOM_vals  - (Array{Float64,1}):
       All the values of the corresponding hyper edges.
-------------------------------------------------------------------------"""
function produce_HOM_tensor(image1_triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
                            image2_triangles::Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1},
                            m::Int; tol::Float64=1e-5,test_mode=false) where {F <: AbstractFloat}
    max_tris  = size(image1_triangles,1)*size(image2_triangles,1)*6

    HOM_indices = Array{Int64,2}(undef,max_tris,3)
    HOM_vals = Array{Float64,1}(undef,max_tris)

    hyperedge_index = 1
    for ((i1,j1,k1),img1_angles) in image1_triangles
        for (indices,img2_angles) in image2_triangles
            for ((i2,j2,k2),angle_perm) in zip(permutations(indices),permutations(img2_angles))

                val = HOM_angle_diff(img1_angles,tuple(angle_perm...),1.0,test_mode=test_mode)
                if val > tol
                    HOM_indices[hyperedge_index,1] = (i1 -1)*m + i2
                    HOM_indices[hyperedge_index,2] = (j1 -1)*m + j2
                    HOM_indices[hyperedge_index,3] = (k1 -1)*m + k2
                    HOM_vals[hyperedge_index] = val
                    hyperedge_index += 1
                end
            end
        end
    end

    hyperedge_index -= 1 #correct last index increment
    return HOM_indices[1:hyperedge_index,:], HOM_vals[1:hyperedge_index]

end

function HOM_angle_diff(angles1::Tuple{F,F,F}, angles2::Tuple{F,F,F}, epsilon::F;test_mode=false) where {F <: AbstractFloat}
    a1,a2,a3 = angles1
    ap1,ap2,ap3 = angles2
    if test_mode
        return ((a1 - ap1)^2+(a2 - ap2)^2+(a3 - ap3)^2) / epsilon^2
    else
        return exp(-((a1 - ap1)^2+(a2 - ap2)^2+(a3 - ap3)^2) / epsilon^2)
    end
end

