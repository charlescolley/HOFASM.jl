function angle(vec1::Union{Array{F,1},Array{Int,1}},vec2::Union{Array{F,1},Array{Int,1}}) where {F <: AbstractFloat}
    acos(dot(vec1, vec2) / (norm(vec1) * norm(vec2))) * 180 / pi
    #TODO: fix DomainError with 1.0000000000000002 problem
end

function vecs_to_triangle(veci::Union{Array{F,1},BitArray{1}},
                          vecj::Union{Array{F,1},BitArray{1}},
                          veck::Union{Array{F,1},BitArray{1}}) where {F <: AbstractFloat}
    seg_ij = vecj - veci
    seg_ik = veck - veci
    seg_jk = veck - vecj

    angle1 = angle(seg_ij, seg_ik)
    angle2 = angle(-seg_ij, seg_jk)
    angle3 = 180.0 - angle1 - angle2
    return (angle1, angle2, angle3)
end

function brute_force_triangles(pts::Array{F,2}) where {F <: AbstractFloat}

    n,d = size(pts)
    # n choose 3 triangles
    triangles = Array{Tuple{Tuple{Int,Int,Int},Tuple{F,F,F}},1}(undef,binomial(n,3))

    index = 1
    for i =1:n
        for j =1:i-1
            for k = 1:j-1
                x = ((k,j,i),vecs_to_triangle(pts[k,:],pts[j,:],pts[i,:]))
                triangles[index] = x
                index += 1
            end
        end
    end

    return triangles
end


function brute_force_triangles(pts::Array{BitArray{1},1}) where {F <: AbstractFloat}

    n = length(pts)
    # n choose 3 triangles
    #TODO: fix F type error
    triangles = Array{Tuple{Tuple{Int,Int,Int},Tuple{Float64,Float64,Float64}},1}(undef,binomial(n,3))

    index = 1
    for i =1:n
        for j =1:i-1
            for k = 1:j-1
                x = ((k,j,i),vecs_to_triangle(pts[k],pts[j],pts[i]))
                triangles[index] = x
                index += 1
            end
        end
    end

    return triangles

end