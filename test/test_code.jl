
#TODO: add in options for the tweaks made to make this testing possible
function test_code()
    n = 30
    m = 30
    source = brute_force_triangles(rand(n,2))
    source[2] = (source[2][1],source[1][2]) #make a duplicate angle
    target = brute_force_triangles(rand(m,2))

    # one triangle for now
    H_indices, H_vals,B_indices,B_vals = build_index_and_bases_tensors(source, target, 5.0)
    #return H_indices, H_vals,B_indices,B_vals
    H_marg_tens = [sym_mode1_marginalization(H_ind,H_vals,n) for (H_ind,H_vals) in zip(H_indices, H_vals)]
    B_marg_tens = [mode1_marginalization(B_ind,B_vals,m) for (B_ind,B_vals) in zip(B_indices, B_vals)]
    HOFASM_marg_ten = numpy_kron(H_marg_tens[1],B_marg_tens[1])

    #HOFASM_perm_marg_ten = Matrix(perm_marginalize2(H_indices[1],B_indices[1],B_vals[1],m))
    HOFASM_perm_marg_ten = HOFASM_mode1_marg(H_indices,B_indices,B_vals,m,n)

    pairs = Make_HOFASM_tensor_pairs(H_indices,B_indices,B_vals,m,n)

    HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-16)
    HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))

   #return HOFASM_perm_marg_ten, HOM_marg_ten
    #return HOM_marg_ten, H_indices, H_vals,B_indices,B_vals
    #check for the same values

    #@assert norm(sort(HOM_vals) - sort(B_vals[1]))/norm(HOM_vals) < 1e-16

    #symmetry check

    @assert norm(HOM_marg_ten - HOM_marg_ten')/norm(HOM_marg_ten) < 1e-14
    println(norm(HOFASM_marg_ten - HOFASM_marg_ten')/norm(HOFASM_marg_ten))
    @assert norm(HOFASM_perm_marg_ten - HOFASM_perm_marg_ten')/norm(HOFASM_perm_marg_ten) < 1e-14

    #equality check
    println(norm(HOM_marg_ten - HOFASM_perm_marg_ten) / norm(HOFASM_perm_marg_ten))
    @assert norm(HOM_marg_ten - HOFASM_perm_marg_ten)/norm(HOFASM_perm_marg_ten) < 1e-14

end

function test_contraction(n::Int=30,m::Int=30,bin_size::Float64=5.0)


    source = brute_force_triangles(rand(n,2))
    #source[2] = (source[2][1],source[1][2]) #make a duplicate angle
    target = brute_force_triangles(rand(m,2))

    # one triangle for now
    H_indices, H_vals,B_indices,B_vals = build_index_and_bases_tensors(source, target, bin_size)
    pairs = Make_HOFASM_tensor_pairs(H_indices,B_indices,B_vals,m,n)
    println("built pairs")

    x = rand(n*m)
    X = Matrix(reshape(x,n,m)')
    Y = rand(n,m)
    preprocessed_pairs = [(findnz(H)[1],findnz(H)[2],B) for (H,B) in pairs]
    #return preprocessed_pairs,X,Y
    t1 = @timed HOFASM_contraction!(preprocessed_pairs,X,Y)
    t1 = t1[2]
    println("HOFASM_contraction ran in $t1")

    implit_marg_kron_y = rand(n*m)
    t2 = @timed implicit_kronecker_model_marginalization_contraction!(B_indices,B_vals,H_indices,x,implit_marg_kron_y,n,m)
    t2 = t2[2]
    println("HOFASM_implicit_marg_kron_contraction ran in $t2")

    #HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-5)
    #HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))
    #t3 = @timed HOM_marg_ten*x
    #t3 = t3[2]
    #println("HOM ran in time $t3")

    return implit_marg_kron_y , Y

    diff = norm(reshape(implit_marg_kron_y,n,m) - Y)/norm(Y)
    println(diff)
    @assert diff < 1e-14
end



#special testing
#@pycall numpy
#function numpy_kron(A,B)
#    return numpy.kron(A,B)
#end


function brute_force_tensor(H_indices,B_indices,B_vals,n,m)

    e1 = size(H_indices,1)
    e2 = size(B_indices,1)

    final_indices = Array{Int,2}(undef,e1*e2,3)
    final_vals = Array{Float64,1}(undef,e1*e2)

    index = 1
    for (i1,i2,i3) in eachrow(H_indices)
        for ((j1,j2,j3),v) in zip(eachrow(B_indices),B_vals)
            final_indices[index,1] = (i1-1)*m + j1
            final_indices[index,2] = (i2-1)*m + j2
            final_indices[index,3] = (i3-1)*m + j3
            final_vals[index] = v
            index += 1
        end
    end

    return final_indices, final_vals
end

function perm_marginalize(H_index,B_indices::Array{Int64,2},B_vals::Array{Float64,1},m::Int)

    z = size(B_indices,1)
    ei = zeros(Int,z*6)
    ej = zeros(Int,z*6)
    vals = zeros(Float64,z*6)
    index = 1

    for p in permutations([1,2,3])

        i1,i2,i3 = H_index[p]

        for (k,val) in zip(1:z,B_vals)

            println(B_indices[k,:])
            println("H_indices: $i1, $i2, $i3")

            (_,j2,j3) = B_indices[k,p]
            ei[index] = (i2-1)*m + j2
            ej[index] = (i3-1)*m + j3
            vals[index] = val
            index += 1

        end
    end

    return sparse(ei,ej,vals,m^2,m^2)
end


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

    return sparse(ei,ej,new_vals,m,m)::SparseMatrixCSC{Float64,Int64}
end

