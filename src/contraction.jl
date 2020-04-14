"""
"""



"""----------------------------------------------------------------------------
    Computes the contraction C(1,x) using the indicies of the tensor
  non-zeros to compute the marginalization step and then the vector contraction
  step after.
----------------------------------------------------------------------------"""
function implicit_kronecker_model_marginalization_contraction!(B_indices::Array{Array{Int,2},1},
                                                               B_vals::Array{Array{Float64,1},1},
                                                               H_indices::Array{Array{Int,2},1},
                                                               x::Array{Float64,1},
                                                               y::Array{Float64,1},
                                                               n::Int,m::Int)

    for i in 1:n
        for j in 1:m
           y[(i-1)*m + j] = 0.0
       end
    end

    for (Hn_indices,Bn_indices,Bn_vals) in zip(H_indices,B_indices,B_vals)

        for i = 1:size(Hn_indices,1)

            for p in permutations([1,2,3])
                _,i2,i3  = Hn_indices[i,p]

                for j = 1:size(Bn_indices,1)
                    _,j2,j3 = Bn_indices[j,p]
                    y[(i2-1)*m + j2] +=  Bn_vals[j]*x[(i3-1)*m + j3]

                end
            end
        end
    end
end

#= TODO: verify this can be retired
function implicit_kronecker_mode1_marginalization(Bn_indices::Array{Int,2},
    Bn_vals::Array{Float64,1},Hn_indices::Array{Int,2},n,m)

    ei = Array{Int,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))
    ej = Array{Int,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))
    vals = Array{Float64,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))

    index = 1
    for i = 1:size(Hn_indices,1)

        i1 = Hn_indices[i,1]
        i2 = Hn_indices[i,2]
        i3 = Hn_indices[i,3]

        for j = 1:size(Bn_indices,1)

            #_,j2,j3 = Bn_indices[j,:]
            j2 = Bn_indices[j,2]
            j3 = Bn_indices[j,3]

            ei[index] = (i2-1)*m + j2
            ej[index] = (i3-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1

            ei[index] = (i3-1)*m + j2
            ej[index] = (i2-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1

            ei[index] = (i1-1)*m + j2
            ej[index] = (i2-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1

            ei[index] = (i2-1)*m + j2
            ej[index] = (i1-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1

            ei[index] = (i1-1)*m + j2
            ej[index] = (i3-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1

            ei[index] = (i3-1)*m + j2
            ej[index] = (i1-1)*m + j3
            vals[index] = Bn_vals[j]
            index += 1
        end
    end

    return sparse(ei,ej,vals,n*m,n*m)
    #return ei,ej,vals,n*m
end
=#

function sym_mode1_marginalization(indices::Array{Int,2},vals::Array{Float64,1},m::Int)
    n,d = size(indices)

    ei = Array{Int,1}(undef,6*length(vals))
    ej = Array{Int,1}(undef,6*length(vals))
    new_vals = Array{Float64,1}(undef,6*length(vals))

    index = 1
    for i =1:n
        i1,i2,i3 = indices[i,:]

        ei[index] = i1
        ej[index] = i2
        new_vals[index] = vals[i]
        index += 1

        ei[index] = i2
        ej[index] = i1
        new_vals[index] = vals[i]
        index += 1

        ei[index] = i2
        ej[index] = i3
        new_vals[index] = vals[i]
        index += 1

        ei[index] = i3
        ej[index] = i2
        new_vals[index] = vals[i]
        index += 1

        ei[index] = i1
        ej[index] = i3
        new_vals[index] = vals[i]
        index += 1

        ei[index] = i3
        ej[index] = i1
        new_vals[index] = vals[i]
        index += 1
    end

    return sparse(ei,ej,new_vals,m,m)
end

function mode1_marginalization(indices::Array{Int,2},vals::Array{Float64,1},m::Int)
    n,d = size(indices)

    ei = Array{Int,1}(undef, length(vals))
    ej = Array{Int,1}(undef,length(vals))
    new_vals = Array{Float64,1}(undef, length(vals))

    index = 1
    for i =1:n

        ei[index] = indices[index,2]
        ej[index] = indices[index,3]
        new_vals[index] = vals[index]
        index += 1

    end

    return sparse(ei,ej,new_vals,m,m)
end


"""----------------------------------------------------------------------------
    Computes the contraction C(1,x) using list of pairs of the marginalized Hn
  Bn pairs. Hn is typically quite sparse, so it's more efficient to figure out
  which rows of the folded multiplicand should be multiplied to Bn, rather
  than doing the explicit HnXBnᵀ operation.
----------------------------------------------------------------------------"""
function HOFASM_contraction!(tensor_pairs::Array{NTuple{2,SparseMatrixCSC{Float64,Int64}},1},
                             X_k::Array{Float64,2},X_k_1::Array{Float64,2}) #where {F <: AbstractFloat}

    n,m = size(X_k_1)

    for i=1:n
       for j=1:m
           X_k_1[i,j] = 0.0
       end
    end

    for (Hn,Bn) in tensor_pairs
#         X_k_1 .+= (Bn*(Hn*X_k)' )'
        is,js,vs = findnz(Hn)
        for (i,j,v) in zip(is,js,vs)
            if v > 1.0
                X_k_1[i,:]  .+= v*Bn*X_k[j,:]
            else
                X_k_1[i,:]  .+= Bn*X_k[j,:]
            end
        end
    end


end

"""----------------------------------------------------------------------------
    Computes the contraction C(1,x) using list of pairs of the marginalized Hn
  Bn pairs. Hn is typically quite sparse, so it's more efficient to figure out
  which rows of the folded multiplicand should be multiplied to Bn, rather
  than doing the explicit HnXBnᵀ operation. The input here is a preprocessed
  version of each of the Hn, so that the findnz operation isn't called
  repeatedly.
----------------------------------------------------------------------------"""
function HOFASM_contraction!(tensor_pairs::Array{Tuple{Array{Int,1},Array{Int,1},Array{Float64,1},
                                                       SparseMatrixCSC{Float64,Int64}},1},
                             X_k::Array{Float64,2},X_k_1::Array{Float64,2})

    n,m = size(X_k_1)
    for i=1:n
       for j=1:m
           X_k_1[i,j] = 0.0
       end
    end

    for (is,js,vs,Bn) in tensor_pairs
        #is,js,_ = findnz(Hn)
        for (i,j,v) in zip(is,js,vs)
            if v > 1.0
                X_k_1[i,:]  .+= v*Bn*X_k[j,:]
            else
                X_k_1[i,:]  .+= Bn*X_k[j,:]
            end
        end
    end


end

"""----------------------------------------------------------------------------
----------------------------------------------------------------------------"""
function HOFASM_contraction!(marginalized_tensors::Array{SparseMatrixCSC{Float64,Int64},1},
                             x_k::Array{Float64,1},x_k_1::Array{Float64,1})


    for i=1:length(x_k_1)
       x_k_1[i] = 0.0
    end

    for Tn in marginalized_tensors
        x_k_1 .+= Tn*x_k
    end

end


#TODO: need to correct the output
#TODO: check to see if this version is faster now that we don't explicitly compute HnX
function SIMD_sparse_matrix_multiplication(Hn_matrices::Array{SparseMatrixCSC{Float64,Int64},1},
                                    Bn_matrices::Array{SparseMatrixCSC{Float64,Int64},1},
                                    X::Array{Float64,2},res_loc::Array{Float64,2})

    #pack the vectors X into groups of 8
    n,m= size(X)
    num_segs = Int(ceil(m/8))
    X_segments = Array{Array{Float64,2},1}(undef,num_segs)

    for i=1:num_segs
        if i == num_segs

            X_segments[i] = X'[:,(i-1)*8+1:end]
        else
            X_segments[i] = X'[:,(i-1)*8+1:i*8]
        end

    end

    packed_segments = [pack(x) for x in X_segments]

    temp_segs = Array{Array{Float64,2},1}(undef,num_segs)

    for (Hn,Bn) in zip(Hn_matrices,Bn_matrices)

        #compute Bn*X'
        temp = zeros(m,n)
        for i=1:num_segs

            if i == num_segs
                temp[:,(i-1)*8+1:end] = unpack(Bn*packed_segments[i])
            else
                temp[:,(i-1)*8+1:i*8] = unpack(Bn*packed_segments[i])
            end

        end

        #repack temp segments
        for i=1:num_segs
            if i == num_segs
                temp_segs[i] = temp'[:,(i-1)*8+1:end]
            else
               temp_segs[i] = temp'[:,(i-1)*8+1:i*8]
            end
        end
        packed_temp_segs = [pack(x) for x in temp_segs]

        #compute A*(BX')' and store in result location
        for i = 1:num_segs
            if i == num_segs
                res_loc[:,(i-1)*8+1:end] = unpack(Hn*packed_temp_segs[i])
            else
                res_loc[:,(i-1)*8+1:i*8] = unpack(Hn*packed_temp_segs[i])
            end
        end
    end


end

"""----------------------------------------------------------------------------
    Computes the marginalization of a 3rd order tensor with respect to a
  generalized transpose operation.
----------------------------------------------------------------------------"""
#allows the application of a permutation before contracting
function perm_mode1_marginalization(indices::Array{Int,2},vals::Array{Float64,1},
                                    perm::Array{Int,1},m::Int)
    n,d = size(indices)

    ei = Array{Int,1}(undef, length(vals))
    ej = Array{Int,1}(undef,length(vals))
    new_vals = Array{Float64,1}(undef, length(vals))

    index = 1
    for i =1:n

        ei[index] = indices[index,perm[2]]
        ej[index] = indices[index,perm[3]]
        new_vals[index] = vals[index]
        index += 1

    end
    index -= 1 #counter the last extra index
    return sparse(ei[1:index],ej[1:index],new_vals[1:index],m,m)::SparseMatrixCSC{Float64,Int64}
end

"""----------------------------------------------------------------------------
    Computes the marginalization of ∑ Hn^{<σ>} ⨂ Bn^{<σ>} from the indices of
  their tensor representations.
----------------------------------------------------------------------------"""
function perm_marginalize(H_indices::Array{Int64,2},B_indices::Array{Int64,2},
                          B_vals::Array{Float64,1},n::Int,m::Int)

    B_nnz = size(B_indices,1)
    H_nnz = size(H_indices,1)

    ei = zeros(Int,B_nnz*H_nnz*6)
    ej = zeros(Int,B_nnz*H_nnz*6)
    vals = zeros(Float64,B_nnz*H_nnz*6)
    index = 1

    for i in 1:H_nnz
        for p in permutations([1,2,3])

            i1,i2,i3 = H_indices[i,p]

            for j in 1:B_nnz

                (_,j2,j3) = B_indices[j,p]
                ei[index] = (i2-1)*m + j2
                ej[index] = (i3-1)*m + j3
                vals[index] = B_vals[j]
                index += 1

            end
        end
    end

    return sparse(ei,ej,vals,n*m,n*m)
end

