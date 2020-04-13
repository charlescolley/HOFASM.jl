
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

    #ei = Array{Int,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))
    #ej = Array{Int,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))
    #vals = Array{Float64,1}(undef,6*size(Hn_indices,1)*length(Bn_vals))
    for (Hn_indices,Bn_indices,Bn_vals) in zip(H_indices,B_indices,B_vals)

        #println(Hn_indices)
        #println(Bn_indices)
        #println(Bn_vals)
        for i = 1:size(Hn_indices,1)

            for p in permutations([1,2,3])
                _,i2,i3  = Hn_indices[i,p]

                for j = 1:size(Bn_indices,1)
                    #println(p)
                    _,j2,j3 = Bn_indices[j,p]
                    #_,j2,j3 = Bn_indices[j,:]
        #                j2 = Bn_indices[j,2]
        #                j3 = Bn_indices[j,3]

                    y[(i2-1)*m + j2] +=  Bn_vals[j]*x[(i3-1)*m + j3]

                end
            end
        end
    end
end


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

function HOFASM_contraction!(tensor_pairs::Array{Tuple{Array{Int,1},Array{Int,1},SparseMatrixCSC{Float64,Int64}},1},
                             X_k::Array{Float64,2},X_k_1::Array{Float64,2}) #where {F <: AbstractFloat}

    n,m = size(X_k_1)
    for i=1:n
       for j=1:m
           X_k_1[i,j] = 0.0
       end
    end

    for (is,js,Bn) in tensor_pairs
        #is,js,_ = findnz(Hn)
        for (i,j) in zip(is,js)
            println(size(Bn))
            println(size(X_k))
            X_k_1[i,:]  .+= Bn*X_k[j,:]
        end
    end


end



#TODO: need to verify output is correct
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