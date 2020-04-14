
using PyCall
#special testing
numpy = pyimport("numpy")
function numpy_kron(A,B)
    return numpy.kron(A,B)
end

ERROR_TOL = 1e-13


#Checks to see if under special parameters the marginalized tensors of HOFASM and HOM are equal.
function test_code()
    n = 10
    m = 10
    source = brute_force_triangles(rand(n,2))
    source[2] = (source[2][1],source[1][2]) #make a duplicate angle
    target = brute_force_triangles(rand(m,2))

    H_indices, H_vals,B_indices,B_vals =
        build_index_and_bases_tensors(source, target, 5.0,test_mode=true)
    HOFASM_perm_marg_ten = HOFASM_mode1_marg(H_indices,B_indices,B_vals,m,n)

    HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-16,test_mode=true)
    HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))

    #symmetry check
    @assert norm(HOM_marg_ten - HOM_marg_ten')/norm(HOM_marg_ten) < ERROR_TOL
    #println(norm(HOFASM_marg_ten - HOFASM_marg_ten')/norm(HOFASM_marg_ten))
    @assert norm(HOFASM_perm_marg_ten - HOFASM_perm_marg_ten')/norm(HOFASM_perm_marg_ten) < ERROR_TOL

    #equality check
    println(norm(HOM_marg_ten - HOFASM_perm_marg_ten) / norm(HOFASM_perm_marg_ten))
    @assert norm(HOM_marg_ten - HOFASM_perm_marg_ten)/norm(HOFASM_perm_marg_ten) < ERROR_TOL

end

#checks to see that all the different forms of HOFASM contractions yield the same values
function test_contraction(n::Int=30,m::Int=30,bin_size::Float64=5.0)


    source = brute_force_triangles(rand(n,2))
    target = brute_force_triangles(rand(m,2))


    H_indices, H_vals,B_indices,B_vals = build_index_and_bases_tensors(source, target, bin_size)
    pairs,pair_timing = @timed Make_HOFASM_tensor_pairs(H_indices,B_indices,B_vals,m,n)

    println("built $(length(pairs)) pairs")

    #--------------------------------unprocessed HXBᵀ------------------------------------------#
    x = ones(n*m)
    X = Matrix(reshape(x,m,n)') #julia defaults to col major formatting
    Y1 = rand(n,m)

    t1 = @timed HOFASM_contraction!(pairs,X,Y1)
    println("HOFASM_contraction runtimes;
            pre-processing: $(pair_timing[1])    contraction op:$(t1[2])")

    #--------------------------------pre-processed HXBᵀ----------------------------------------#
    Y2 = rand(n,m)
    preprocessed_pairs,processing_time = @timed [(findnz(H)...,B) for (H,B) in pairs]
    #preprocessed_pairs = [(findnz(H)...,B) for (H,B) in pairs]
    #return preprocessed_pairs,X,Y
    t2 = @timed HOFASM_contraction!(preprocessed_pairs,X,Y2)
    println("preprocessed HOFASM_contraction runtimes;
             pre-processing:$(processing_time[1]+pair_timing[1]) contraction op:$(t2[2])  ")

    #-----------------------------implicit marginalization-------------------------------------#
    implicit_marg_kron_y = rand(n*m)
    t3 = @timed implicit_kronecker_model_marginalization_contraction!(B_indices,B_vals,H_indices,
                                                                      x,implicit_marg_kron_y,n,m)
    println("explicit marg kron HOFASM_contraction runtimes;
             contraction op:$(t3[2])")

    #-----------------------------explicit marginalization-------------------------------------#

    kron_pairs, kron_times = @timed [perm_marginalize(Hn_ind,Bn_ind,Bn_val,n,m) for (Bn_ind,Bn_val,Hn_ind) in zip(B_indices,B_vals,H_indices)]
    explicit_marg_kron_y = rand(n*m)

    t4 = @timed HOFASM_contraction!(kron_pairs,x,explicit_marg_kron_y)
    println("explicit marg kron HOFASM_contraction runtimes;
             pre-processing:$(kron_times[1]) contraction op:$(t4[2])  ")

    #HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-5)
    #HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))
    #t3 = @timed HOM_marg_ten*x
    #t3 = t3[2]
    #println("HOM ran in time $t3")

  #  return Matrix(reshape(implicit_marg_kron_y,m,n)'), Y

    diff1 = norm(Y1 - Y2)/norm(Y2)
    println(diff1)
    @assert diff1 < ERROR_TOL

    diff2 = norm(Matrix(reshape(implicit_marg_kron_y,m,n)') - Y2)/norm(Y2)
    println(diff2)
    @assert diff2 < ERROR_TOL


    diff3 = norm(Matrix(reshape(explicit_marg_kron_y,m,n)') - Y2)/norm(Y2)
    println(diff3)
    @assert diff3 < ERROR_TOL
    println("performance ratio impli:new = $(t3[2]/t2[2])")

end





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


function HOFASM_mode1_marg(H_indices::Array{Array{Int64,2},1},B_indices::Array{Array{Int64,2},1},
                            B_vals::Array{Array{Float64,1},1},m::Int,n::Int)

        return sum([sum(
                [numpy_kron(perm_mode1_marginalization(H_idx,ones(size(H_idx,1)),p,n),
                            perm_mode1_marginalization(B_idx,v,p,m)
                            )
                            for p in permutations((1,2,3))
                ]) for (H_idx,B_idx,v) in zip(H_indices,B_indices,B_vals)])
end