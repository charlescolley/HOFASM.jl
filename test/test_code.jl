
using PyCall
#special testing
numpy = pyimport("numpy")
function numpy_kron(A,B)
    return numpy.kron(A,B)
end

ERROR_TOL = 1e-13


#TODO: add in options for the tweaks made to make this testing possible
function test_code()
    n = 10
    m = 10
    source = brute_force_triangles(rand(n,2))
    source[2] = (source[2][1],source[1][2]) #make a duplicate angle
    target = brute_force_triangles(rand(m,2))

    H_indices, H_vals,B_indices,B_vals =
        build_index_and_bases_tensors(source, target, 5.0,test_mode=true)

    #H_marg_tens = [sym_mode1_marginalization(H_ind,H_vals,n) for (H_ind,H_vals) in zip(H_indices, H_vals)]
    #B_marg_tens = [mode1_marginalization(B_ind,B_vals,m) for (B_ind,B_vals) in zip(B_indices, B_vals)]
    #HOFASM_marg_ten = numpy_kron(H_marg_tens[1],B_marg_tens[1])

    #HOFASM_perm_marg_ten = Matrix(perm_marginalize2(H_indices[1],B_indices[1],B_vals[1],m))
    HOFASM_perm_marg_ten = HOFASM_mode1_marg(H_indices,B_indices,B_vals,m,n)

    #pairs = Make_HOFASM_tensor_pairs(H_indices,B_indices,B_vals,m,n)

    HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-16,test_mode=true)
    HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))

    #return HOFASM_perm_marg_ten, HOM_marg_ten
    #return HOM_marg_ten, H_indices, H_vals,B_indices,B_vals
    #check for the same values

    #@assert norm(sort(HOM_vals) - sort(B_vals[1]))/norm(HOM_vals) < 1e-16

    #symmetry check

    @assert norm(HOM_marg_ten - HOM_marg_ten')/norm(HOM_marg_ten) < ERROR_TOL
    #println(norm(HOFASM_marg_ten - HOFASM_marg_ten')/norm(HOFASM_marg_ten))
    @assert norm(HOFASM_perm_marg_ten - HOFASM_perm_marg_ten')/norm(HOFASM_perm_marg_ten) < ERROR_TOL

    #equality check
    println(norm(HOM_marg_ten - HOFASM_perm_marg_ten) / norm(HOFASM_perm_marg_ten))
    @assert norm(HOM_marg_ten - HOFASM_perm_marg_ten)/norm(HOFASM_perm_marg_ten) < ERROR_TOL

end

function test_contraction(n::Int=30,m::Int=30,bin_size::Float64=5.0)


    source = brute_force_triangles(rand(n,2))
    #source[2] = (source[2][1],source[1][2]) #make a duplicate angle
    target = brute_force_triangles(rand(m,2))

    # one triangle for now
    H_indices, H_vals,B_indices,B_vals = build_index_and_bases_tensors(source, target, bin_size)
   # return H_indices, H_vals,B_indices,B_vals, source, target
    pairs = Make_HOFASM_tensor_pairs(H_indices,B_indices,B_vals,m,n)

    println("built $(length(pairs)) pairs")

    x = ones(n*m)
    X = Matrix(reshape(x,m,n)') #julia defaults to col major formatting
    Y1 = rand(n,m)

    t1 = @timed HOFASM_contraction!(pairs,X,Y1)
    println("HOFASM_contraction ran in $(t1[2])")


    Y2 = rand(n,m)
    preprocessed_pairs = [(findnz(H)...,B) for (H,B) in pairs]
    #preprocessed_pairs = [(findnz(H)...,B) for (H,B) in pairs]
    #return preprocessed_pairs,X,Y
    t2 = @timed HOFASM_contraction!(preprocessed_pairs,X,Y2)
    println("preprocessed HOFASM_contraction ran in $(t2[2])")


    implit_marg_kron_y = rand(n*m)
    t3 = @timed implicit_kronecker_model_marginalization_contraction!(B_indices,B_vals,H_indices,x,implit_marg_kron_y,n,m)
    println("HOFASM_implicit_marg_kron_contraction ran in $(t3[2])")

    #HOM_indices, HOM_vals = produce_HOM_tensor(source,target,m,tol=1e-5)
    #HOM_marg_ten = Matrix(sym_mode1_marginalization(HOM_indices,HOM_vals,n*m))
    #t3 = @timed HOM_marg_ten*x
    #t3 = t3[2]
    #println("HOM ran in time $t3")

  #  return Matrix(reshape(implit_marg_kron_y,m,n)'), Y
    diff1 = norm(Y1 - Y2)/norm(Y2)
    println(diff1)
    @assert diff1 < ERROR_TOL

    diff2 = norm(Matrix(reshape(implit_marg_kron_y,m,n)') - Y2)/norm(Y2)
    println(diff2)
    @assert diff2 < ERROR_TOL

    println("performance ratio impli:new = $(t2[2]/t1[2])")

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