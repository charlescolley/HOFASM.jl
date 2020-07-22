

function  Bistochastic_Normalization!(X::Array{Float64,2},iterations::Int)

    n,m = size(X)

    for _ in 1:iterations
        for i = 1:n
            X[i,:] ./= norm(X[i,:],1)
        end

        for j = 1:m
            X[:,j] ./= norm(X[:,j],1)
        end
    end

end

function Bistochastic_Normalization!(x::Array{Float64,1},iterations::Int,n::Int,m::Int)

    X = reshape(x,m,n)

    for _ in 1:iterations
        for i = 1:n
            X[i,:] ./= norm(X[i,:],1)
        end

        for j = 1:m
            X[:,j] ./= norm(X[:,j],1)
        end
    end

end

function build_assignment(X::Array{Float64,2};use_colsort::Bool=false,
                          return_matrix=false)
    """-------------------------------------------------------------------------
        For a HOFASM_iteration matrix result X, computing a greedy matching on
      the maximum elements of the columns sequentially. if using the option
      use_colsort, will find the matching in the columns ordered by the largest
      element of each column. If return_matrix is true, a matrix is returned of
      the same dimensions as X, but X(i,j) =  1 iff i is matched to j.

      Notes:
      ------
      * Currently implemented for square matrices
    -------------------------------------------------------------------------"""
    n, m = size(X)
    selected_vertices = Set{Int}()
    matching = Array{Tuple{Int,Int},1}(undef,minimum((n,m)))
    index = 1

    if use_colsort
        col_order = sortperm([maximum(X[:,i]) for i in 1:m],rev=true)
    else
        col_order = 1:m
    end

    for j in col_order #search over columns
        max_val = -Inf
        arg_max = -1
        for i = 1:n
            if (X[i,j] > max_val) && !(i in selected_vertices)
                max_val = X[i,j]
                arg_max = i
            end
        end
        matching[index] = (j,arg_max)
        index += 1
        push!(selected_vertices,arg_max)
    end
    if return_matrix
        Z = zeros(n,m)
        for (i,j) in matching
            Z[i,j] = 1
        end
        return Z
    else
        return matching
    end
end



function HOFASM_iterations(Bn_indices_list::Array{Array{Int,2},1},Bn_vals_list::Array{Array{Float64,1},1},
                           Hn_indices_list::Array{Array{Int,2},1},n::Int,m::Int,max_iterations::Int=12)

    x_k::Array{Float64,1} = rand(Float64,n*m)
    x_k ./= norm(x_k)
    beta = 30 # constant from the paper

    i = 1
    x_k_1::Array{Float64,1} = zeros(n*m)

    while true


        implicit_kronecker_model_marginalization_contraction!(Bn_indices_list,Bn_vals_list,
                                                              Hn_indices_list,x_k,x_k_1,n,m)

        max_elem = maximum(x_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,x_k_1,x_k_1)


        while true

            x_normalized = copy(x_k_1)
            Bistochastic_Normalization!(x_normalized,15,n,m)
  #          print(f"{norm(X_normalized - X_k_1,'fro')}")
            if norm(x_normalized - x_k_1) < 1e-4
                x_k_1 = x_normalized
                break
            else
                x_k_1 = copy(x_normalized)
            end
        end
#        return X_k_1

        x_k_1 ./= norm(x_k_1)
        res = norm(x_k_1 - x_k) / norm(x_k)

        #print("Finished iteration $i with difference $res")
        i += 1
        if i > max_iterations
            return x_k_1
        end

        if res < 1e-4
            return x_k_1
        else
            x_k = copy(x_k_1)
        end
    end
end



function HOFASM_iterations(matrices::Array{SparseMatrixCSC{Float64,Int64},1},n::Int,m::Int,max_iterations::Int=12)

    #    X_k = np.ones((n,m))
    x_k::Array{Float64,1} = rand(Float64,n*m)
    x_k ./= norm(x_k)
    beta = 30 # constant from the paper

    i = 1

    while true

        x_k_1::Array{Float64,1} = zeros(n*m)
        for Tn in matrices
            x_k_1 .+= Tn*x_k
        end

        max_elem = maximum(x_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,x_k_1,x_k_1)


        while true

            x_normalized = copy(x_k_1)
            Bistochastic_Normalization!(x_normalized,15,n,m)
  #          print(f"{norm(X_normalized - X_k_1,'fro')}")
            if norm(x_normalized - x_k_1) < 1e-4
                x_k_1 = x_normalized
                break
            else
                x_k_1 = copy(x_normalized)
            end
        end
#        return X_k_1

        x_k_1 ./= norm(x_k_1)
        res = norm(x_k_1 - x_k) / norm(x_k)

        #print("Finished iteration $i with difference $res")
        i += 1
        if i > max_iterations
            return x_k_1
        end

        if res < 1e-4
            println("tolerance met")
            return x_k_1
        else
            x_k = copy(x_k_1)
        end
    end
end


function HOFASM_iterations(tensor_pairs::Array{NTuple{2,SparseMatrixCSC{Float64,Int64}},1},n::Int,
                           m::Int,max_iterations::Int=12,tol=1e-4)

    #    X_k = np.ones((n,m))
    X_k::Array{Float64,2} = rand(Float64,n,m)
    X_k ./= norm(X_k)
    beta = 30 # constant from the paper

    X_k_1 = Array{Float64,2}(undef,n,m)
    i = 1

    preprocessed_pairs = [(findnz(H)...,B) for (H,B) in tensor_pairs]

    while true

  #      HOFASM_contraction!_test(tensor_pairs,X_k,X_k_1)
        HOFASM_contraction!(preprocessed_pairs,X_k,X_k_1)

 #       SIMD_sparse_matrix_multiplication(Hn_matrices,Bn_matrices,X_k,X_k_1)
#        packed_X_k = pack(X_k')
 #       for (Hn,Bn) in tensor_pairs

#            X_k_1 .+= Hn*(Bn*X_k')'
#            X_k_1 .+= (Bn*(Hn*X_k)' )'

#            temp = unpack(Bn*packed_X_k)'
#            X_k_1 .+= unpack(Hn*pack(temp))
           #println(typeof(Hn*(Bn*X_k')'))
           # X_k_1 .+= Hn*(X_k*Bn')
  #      end

 #       println("done")

        max_elem = maximum(X_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,X_k_1,X_k_1)

        while true

            X_normalized = copy(X_k_1)
            Bistochastic_Normalization!(X_normalized,15)
  #          println("BN")
   #         print("{norm(X_normalized - X_k_1,'fro')}")
            if norm(X_normalized - X_k_1) < 1e-4
                X_k_1 = X_normalized
                break
            else
                X_k_1 = copy(X_normalized)
            end
        end
#        return X_k_1

        X_k_1 ./= norm(X_k_1)
        res = norm(X_k_1 - X_k) / norm(X_k)

  #      println("Finished iteration $i with difference $res")
        i += 1
        if i > max_iterations
            return X_k_1
        end

        if res < tol
            return X_k_1
        else
            X_k = copy(X_k_1)
        end
    end
end


#TODO: could generalize to take a contraction function to unify all graduated assignment routines
function HOM_graduated_assignment(marg_tensor::SparseMatrixCSC{Float64,Int64},max_iterations::Int=12,tol::Float64=1e-4)

    #    X_k = np.ones((n,m))
    n = size(marg_tensor,1)
    m = Int(sqrt(n))
    x_k::Array{Float64,1} = rand(Float64,n)
    x_k ./= norm(x_k)
    beta = 30 # constant from the paper
    i = 1

    while true

        x_k_1 = marg_tensor*x_k

        max_elem = maximum(x_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,x_k_1,x_k_1)

        while true

            x_normalized = copy(x_k_1)
            Bistochastic_Normalization!(x_normalized,10,m,m)
  #          print(f"{norm(X_normalized - X_k_1,'fro')}")
            if norm(x_normalized - x_k_1) < 1e-4
                x_k_1 = x_normalized
                break
            else
                x_k_1 = copy(x_normalized)
            end
        end
#        return X_k_1

        x_k_1 ./= norm(x_k_1)
        res = norm(x_k_1 - x_k) / norm(x_k)

        #print("Finished iteration $i with difference $res")
        i += 1
        if i > max_iterations
            return x_k_1
        end

        if res < 1e-4
            println("tolerance met")
            return x_k_1
        else
            x_k = copy(x_k_1)
        end
    end

end

#Sequential Second-order expansion for higher order matching
function SSoEfHOM(tensor_pairs::Array{NTuple{2,SparseMatrixCSC{Float64,Int64}},1},n::Int,
                  m::Int, max_iterations::Int=12,tol::Float64=1e-4)


    X_k::Array{Float64,2} = rand(Float64,n,m)
    beta = 30 # constant from the paper

    X_k_1 = Array{Float64,2}(undef,n,m)
    B_k_1 = Array{Float64,2}(undef,n,m)
    temp2 = Array{Float64,2}(undef,n,m)
    i = 1

    preprocessed_pairs = [(findnz(H)...,B) for (H,B) in tensor_pairs]

    HOFASM_contraction!(preprocessed_pairs,X_k,X_k_1)
    S_opt = dot(X_k,X_k_1)
    x_opt = build_assignment(rand(n,m),return_matrix=true)


    while true

        #Graduated assignment routine
        max_elem = maximum(X_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,B_k_1,X_k_1)

        while true

            X_normalized = copy(B_k_1)
            Bistochastic_Normalization!(X_normalized,15)

            if norm(X_normalized - B_k_1) < 1e-6
                B_k_1 = X_normalized
                break
            else
                B_k_1 = copy(X_normalized)
            end
        end

        B_k_1 = build_assignment(B_k_1,return_matrix=true)

        diff = B_k_1-X_k_1
        println(norm(diff))
        C = dot(X_k_1,diff)
    #    println(C)


        HOFASM_contraction!(preprocessed_pairs,diff,temp2)
        D = dot(diff,temp2)
#        println(D)


        if D >= 0
            X_k = copy(B_k_1)
        else
            r = minimum((-C/D,1))
            X_k .+= r * diff
        end

        temp3 = copy(B_k_1)
        HOFASM_contraction!(preprocessed_pairs,B_k_1,temp3)
        S = dot(temp3,B_k_1)
        println(S)

        if S >= S_opt
            x_opt = copy(B_k_1)
            S_opt = S
        end


  #      println(X_k)

        i += 1
        if i >= max_iterations
            break
        else
            #tensor vector contraction
            HOFASM_contraction!(preprocessed_pairs,X_k,X_k_1)
        end
    end

    return x_opt
end

#=
#TODO: could generalize to take a contraction function to unify all graduated assignment routines
function HOM_graduated_assignment(A::ssten.COOTen,max_iterations::Int=12,tol::Float64=1e-4)

    #    X_k = np.ones((n,m))
    n = A.cubical_dimension
    m = Int(sqrt(n))
    x_k::Array{Float64,1} = rand(Float64,n)
    x_k ./= norm(x_k)
    beta = 30 # constant from the paper
    i = 1

    while true

        x_k_1 = ssten.contract_k_1(A,x_k)

        max_elem = maximum(x_k_1)
        inflation_normalization = x -> exp(x*beta/max_elem)
        broadcast!(inflation_normalization,x_k_1,x_k_1)

        while true

            x_normalized = copy(x_k_1)
            Bistochastic_Normalization!(x_normalized,10,m,m)
  #          print(f"{norm(X_normalized - X_k_1,'fro')}")
            println(norm(x_normalized - x_k_1))
            if norm(x_normalized - x_k_1) < 1e-4
                x_k_1 = x_normalized
                break
            else
                x_k_1 = copy(x_normalized)
            end
        end
#        return X_k_1

        x_k_1 ./= norm(x_k_1)
        res = norm(x_k_1 - x_k) / norm(x_k)

        #print("Finished iteration $i with difference $res")
        i += 1
        if i > max_iterations
            return x_k_1
        end

        if res < 1e-4
            println("tolerance met")
            return x_k_1
        else
            x_k = copy(x_k_1)
        end
    end

end
=#