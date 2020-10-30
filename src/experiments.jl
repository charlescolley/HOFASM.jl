"""-----------------------------------------------------------------------------
  This routine is the example in ImageFeatures.jl modified to use the HOFASM 
  method and diplay the matchings between the points. This method is primarily 
  included to provide a template for a visual interface for future 
  experimentation. 

-----------------------------------------------------------------------------"""
function align_photos(img_1,img_2)

    img1 = load(img_1)
    img2 = load(img_2)

    #grayscale for fast corners
    img1 = Gray.(img1)
    img2 = Gray.(img2)

    #------#  Get image features

    orb_params = ORB(num_keypoints = 100)
    #brief_params = BRIEF(size = 256, window = 10, seed = 123)

    keypoints_1 = Keypoints(fastcorners(img1, 12, 0.4))
    keypoints_2 = Keypoints(fastcorners(img2, 12, 0.4))

    desc_1, ret_keypoints_1 = create_descriptor(img1, orb_params)
    desc_2, ret_keypoints_2 = create_descriptor(img2, orb_params)
    #------#:from imageFeatures.jl demo
    n = length(desc_1)
    m = length(desc_2)

    source_triangles = brute_force_triangles(desc_1)
    target_triangles = brute_force_triangles(desc_2)

    #build tensors
    index_tensor_indices, index_tensor_vals , bases_tensor_indices, bases_tensor_vals =
    build_index_and_bases_tensors(source_triangles, target_triangles, 5.0)

    marg_ten_pairs, kron_time =
        @timed Make_HOFASM_tensor_pairs(index_tensor_indices,bases_tensor_indices,
                                        bases_tensor_vals,n,m)

    x, iteration_time = @timed HOFASM_iterations(marg_ten_pairs,n,m)

    matching = build_assignment(x,use_colsort=true)
    kp_matchings = [[ret_keypoints_1[j],ret_keypoints_2[i]] for (i,j) in matching]

    grid = draw_matches(img1, img2, kp_matchings)

    imshow(grid)
    return grid
end

"""-----------------------------------------------------------------------------
  This is a routine works to run the HOFASM alignment method on a pair of source
  and target embeddings. This method is primarily included to facilitate 
  additional experimentation. 

-----------------------------------------------------------------------------"""
function align_embeddings(source::Array{Float64,2},target::Array{Float64,2};method="HOFASM")

    n = size(source,1)
    m = size(target,1)

    source_triangles = brute_force_triangles(source)
    target_triangles = brute_force_triangles(target)


    if method == "HOFASM"
        #build tensors
        index_tensor_indices, index_tensor_vals , bases_tensor_indices, bases_tensor_vals =
        build_index_and_bases_tensors(source_triangles, target_triangles, 5.0)

        marg_ten_pairs, kron_time =
            @timed Make_HOFASM_tensor_pairs(index_tensor_indices,bases_tensor_indices,
                                            bases_tensor_vals,n,m)

        x, iteration_time = @timed HOFASM_iterations(marg_ten_pairs,n,m)

    elseif method == "HOM"
        indices, vals = produce_HOM_tensor(source_triangles,target_triangles,m,tol=1e-12)
        marginalized_tensor = sym_mode1_marginalization(indices,vals,n*m)
        x =  Array(reshape(HOM_graduated_assignment(marginalized_tensor,m),n,m)')
    else
        raise()
    end

    return x, build_assignment(x,use_colsort=true)
end

"""-----------------------------------------------------------------------------
  Serialized timing experiments for the HOFASM and HOM methods. Experiments vary
  over different sizes of graphs with a fixed noise level (sigma = 1). 

  Inputs
  ------
  * trials - (Int):
     How many trials to run. Extra trial is added and skipped to account
     for compile time. 
  * seed - (Bool):
     Whether or not to seed the experiments. Seeds are generated seeding Random.jl
     with 0, and generating as many seeds as there are experiments.
     seeds. 
  * method - (String):
     Whether to run 'HOFASM' or 'HOM' for the experiment. sub_method is passed to 
     the synthetic_HOFASM to determine which type of marginalization to use. 
  
  Outputs
  -------
  * Experiments - (Dict{Int,Array{1,Floats}}):
    A dictionary linking each of the experiment's n value linked to the runtimes
    of the trials run. 
-----------------------------------------------------------------------------"""
function timing_experiments(trials = 100,seed=false;method="HOFASM",sub_method="new")
 #   kron_formation_times = []
    run_times = []
    n_values = [10,20,30,40,60,80,100]
    Experiments = Dict()
    sigma = 1.0

    if seed
        Random.seed!(0)
        seeds = rand(UInt64, length(n_values), trials+1)
    end

    n_exp_j = 1
    for n in n_values

        kron_results = []
        runtime_results = []
#        kron_time = 0.0
        run_time = 0.0
        for i in 1:trials+1
            if method == "HOFASM"
                _ , kt, rt, _ = synthetic_HOFASM(n,sigma,method=sub_method,seed = seeds[n_exp_j,i])
            elseif method == "HOM"
                kt = 0.0
                _, rt, _ = synthetic_HOM(n,sigma,seed = seeds[n_exp_j,i])
            end

            if i == 0
                continue #skip due to compile time
            end
            push!(runtime_results,rt+kt)

        end
        Experiments[n] = runtime_results
             n_exp_j += 1

        println("finished problem size $n")
    end

    return Experiments
end


"""-----------------------------------------------------------------------------
  This function runs timing experiments using the Distributed.jl package. 
  Experiments vary over different sizes of graphs with a fixed noise 
  level (sigma = 1).

Inputs
------
* nprocs - (Int):
   How many processes have been allocated.
* trials - (Int):
   How many trials to run. Extra trial is added and skipped to account
   for compile time. 
* seed - (Bool):
   Whether or not to seed the experiments. Seeds are generated seeding Random.jl
   with 0, and generating as many seeds as there are experiments. Lists of seeds
   for each processer are generated, so the number of processes may affect final 
   seeds. 
* method - (String):
   Whether to run 'HOFASM' or 'HOM' for the experiment. sub_method is passed to 
   the synthetic_HOFASM to determine which type of marginalization to use. 

Outputs
-------
* Experiments - (Dict{Int,Array{1,Floats}}):
  A dictionary linking each of the experiment's n value linked to the runtimes
  of the trials run. 

-----------------------------------------------------------------------------"""
function distributed_timing_experiments(nprocs::Int,trials = 100,seed=false;method="HOFASM",sub_method="new")

    @assert average_over % nprocs == 0 #must divide for simplicitly

    @everywhere include_string(Main,$(read("HOFASM.jl",String)),"HOFASM.jl")
 #   kron_formation_times = []

    n_values = [10,20,30,40,60,80,100]

    trials_per_process = Int(trials/nprocs)

    if seed
        Random.seed!(0)
        seeds = rand(UInt64, length(n_values),nprocs,trials_per_process + 1)
    end


    Experiments = Dict()
    sigma = 1.0

    #
    #  trials computed on each process
    #
    function sub_tests(trials,n,sigma,method,sub_method;seeds::Array{UInt,1}=nothing)

        if seeds !== nothing
            @assert length(seeds) == trials+1
        end
        results = []
        for i in 1:trials + 1

            if method == "HOFASM"
                _ , kt, rt, _  = synthetic_HOFASM(n,sigma,method=sub_method;seed = seeds[i])
                #_ , kt, rt, _ = synthetic_HOFASM(n,sigma,method=sub_method)
            elseif method == "HOM"
                kt = 0.0
                _, rt, _  = synthetic_HOM(n,sigma;seed = seeds[i])
            end

            if i == 1
                continue #skip run with compile time
            else
                push!(results,kt+rt)
            end

        end
        return results
    end

    n_exp_j = 1
    for n in n_values

        runtime_results = []
        futures = []

        for p in 2:nprocs +1  #leave process 1 as the parent process
            f = @spawnat p sub_tests(trials_per_process,n,sigma,method,sub_method;seeds = seeds[n_exp_j,p-1,:])
            push!(futures,f)
        end


        runtime_results = []
        for future in futures
            r = fetch(future)
            runtime_results = [runtime_results; r] #concatonate on
        end

        Experiments[n] = runtime_results
        n_exp_j += 1
        println("finished problem size $n")
    end

#    kron_formation_times,
    return Experiments
end

"""-----------------------------------------------------------------------------
  Serialized accuracy experiments for the HOFASM and HOM methods. Experiments 
  vary over different noise levels (sigma) with a fixed noise level (n = 30).
   
Inputs
------
* trials - (Int):
   How many trials to average over. 
* seed - (Bool):
   Whether or not to seed the experiments. Seeds are generated seeding Random.jl
   with 0, and generating as many seeds as there are experiments. Lists of seeds
   for each processer are generated, so the number of processes may affect final 
   seeds. 
* method - (String):
   Whether to run 'HOFASM' or 'HOM' for the experiment. sub_method is passed to 
   the synthetic_HOFASM to determine which type of marginalization to use. 

Outputs
-------
* Experiments - (Dict{Int,Array{1,Floats}}):
  A dictionary linking each of the experiment's sigma value linked to the 
  runtimes of the trials run. 

-----------------------------------------------------------------------------"""
function accuracy_experiments(trials::Int = 2,seed=false;method="HOFASM",sub_method="new")

    sigmas = [.025,.05,.0725,.1,.125,.15,.175,.2]

    Experiments = Dict()

    n = 30

    if seed
        Random.seed!(0)
        seeds = rand(UInt64,length(sigmas), trials)
    else
        exp_seed = nothing
    end

    sigma_exp_j = 1
    for sigma in sigmas

        colsort_accuracies = []

        for i in 1:trials
            if seed
                exp_seed = seeds[sigma_exp_j,i]
            end

            if method == "HOFASM"
                X , _, _, p  = synthetic_HOFASM(n,sigma,method=sub_method;seed=exp_seed)
            elseif method == "HOM"
                X,_, p  = synthetic_HOM(n,sigmal;seed=seeds[sigma_exp_j,i])
            end

            assignment = build_assignment(X,use_colsort=true)

            colsort_accuracy = sum([1.0 for (i,j) in assignment if p[i] == j])/n

            push!(colsort_accuracies,colsort_accuracy)
        end

        Experiments[sigma] = colsort_accuracies
        sigma_exp_j += 1
    end

    return Experiments
end

"""-----------------------------------------------------------------------------
  This function runs timing experiments using the Distributed.jl package. 
  Experiments vary over different noise levels (sigma) with a fixed noise level 
  (n = 30).

Inputs
------
* trials - (Int):
   How many trials to run. Extra trial is added and skipped to account
   for compile time. 
* seed - (Bool):
   Whether or not to seed the experiments. Seeds are generated seeding Random.jl
   with 0, and generating as many seeds as there are experiments. Lists of seeds
   for each processer are generated, so the number of processes may affect final 
   seeds. 
* method - (String):
   Whether to run 'HOFASM' or 'HOM' for the experiment. sub_method is passed to 
   the synthetic_HOFASM to determine which type of marginalization to use. 

Outputs
-------
* Experiments - (Dict{Int,Array{1,Floats}}):
  A dictionary linking each of the experiment's n value linked to the runtimes
  of the trials run. 

-----------------------------------------------------------------------------"""
function distributed_accuracy_experiments(average_over::Int = 2,seed::Bool=false;method="HOFASM",sub_method="new")

    @everywhere include_string(Main,$(read("HOFASM.jl",String)),"HOFASM.jl")
    sigmas = [.025,.05,.0725,.1,.125,.15,.175,.2]

    if seed
        Random.seed!(0)
        seeds = rand(UInt64, length(sigmas), average_over)
    else
        seed = nothing
    end

    Experiments = Dict()

    n = 30

    sigma_exp_j = 1
    for sigma in sigmas

        futures = []
#        kron_time = 0.0

        for i in 1:average_over

            if seed
                exp_seed = seeds[sigma_exp_j,i]
            end

            if method == "HOFASM"
                f = @spawn synthetic_HOFASM(n,sigma,method=sub_method;seed=exp_seed)
            elseif method == "HOM"
                f = @spawn synthetic_HOM(n,sigma;seed=seeds[sigma_exp_j,i])
            end
            push!(futures,f)
            #transpose is need to maintain row major format
        end

        colsort_accuracies = []
        for future in futures
            if method == "HOFASM"
                X , _, _, p  = fetch(future)
            elseif method == "HOM"
                X, _, p  = fetch(future)
            end

            assignment = build_assignment(X,use_colsort=true)
            colsort_accuracy = sum([1.0 for (i,j) in assignment if p[i] == j])/n

            push!(colsort_accuracies,colsort_accuracy)
        end

        Experiments[sigma] = colsort_accuracies
        sigma_exp_j += 1
    end

    return Experiments
end

function outlier_experiments(average_over::Int = 2;method="HOFASM")

    sigma = .1
    max_outliers = 20

    colsort_accuracies = []
    colsort_variances = []

    n = 20

    for outliers = 0:2:max_outliers

#        kron_time = 0.0
        test_argsort_accuracies = []
        test_colsort_accuracies = []

        for i in 1:average_over
            if method =="HOM"
                X , _, p = synthetic_HOM(n,sigma,outliers)
            else
                X , _, _, p = synthetic_HOFASM(n,sigma,outliers)
            end

            assignment = build_assignment(X,use_colsort=true)
            colsort_accuracy = sum([1.0 for (i,j) in assignment if p[i] == j])/(n+outliers)

            push!(test_colsort_accuracies,colsort_accuracy)
        end


        push!(colsort_variances,var(test_colsort_accuracies))
        push!(colsort_accuracies,sum(test_colsort_accuracies)/average_over)
    end

    return colsort_variances, colsort_accuracies
end

function scaling_experiments(average_over::Int = 2; method ="HOFASM")

    sigma = .05
    outliers = 5
    scalings = range(1,stop=3,length=10)

    colsort_accuracies = []
    colsort_variances = []

    n = 20

    for scale in scalings

        test_colsort_accuracies = []

        for i in 1:average_over

            if method =="HOM"
                X , _, p = synthetic_HOM(n,sigma,outliers,scale)
            else
                X , _, _, p = synthetic_HOFASM(n,sigma,outliers,scale)
            end

            assignment = build_assignment(X,use_colsort=true)
            colsort_accuracy = sum([1.0 for (i,j) in assignment if p[i] == j])/(n+outliers)

            push!(test_colsort_accuracies,colsort_accuracy)

        end


        push!(colsort_variances,var(test_colsort_accuracies))
        push!(colsort_accuracies,sum(test_colsort_accuracies)/average_over)
    end


    return colsort_variances, colsort_accuracies
end


#=       ---------------Currently not functioning-----------------

# displays the matching results of the source and target points in blue and red respectively
# outliers have a reduced alpha value
function display_synthetic_result(source,target,n,matching)

#    fig = figure("pyplot_scatterplot",figsize=(10,10))
#    ax = PyPlot.axes()




    PyPlot.scatter(source[1:n,1],source[1:n,2],c = "blue")
    PyPlot.scatter(source[n:end,1],source[n:end,2],c = "blue",alpha=0.6)

    PyPlot.scatter(target[1:n,1],target[1:n,2],c = "green")
    PyPlot.scatter(target[n:end,1],target[n:end,2],c = "green",alpha=0.6)


    for (i,j) in matching
        if i == j
            PyPlot.plot(source[i,:], target[j,:],c="black")
        else
            PyPlot.plot(source[i,:], target[j,:],c="red")
        end
    end

    PyPlot.show()
end

function run_and_display_synthetic_exp(n,sigma,outliers,scale)

    X, _,_,source_points, target_points = synthetic_problem_new(n,sigma,outliers,scale)
    display_synthetic_result(source_points,target_points,n,build_assignment(X,use_colsort=true))

end

=#


# Helper Functions from
#    https://www.juliabloggers.com/image-stitching-part-2/

# this function takes the two images and concatenates them horizontally.
# to horizontally concatenate, both images need to be made the same
# vertical size
function pad_display(img1, img2)
    img1h = size(img1, 1)
    img2h = size(img2, 1)
    mx = max(img1h, img2h);
    hcat(vcat(img1, zeros(RGB{Float64},
                max(0, mx - img1h), size(img1, 2))),
        vcat(img2, zeros(RGB{Float64},
                max(0, mx - img2h), size(img2, 2))))
end

function draw_matches(img1, img2, matches)
    # instead of having grid = [img1 img2], we'll use the new
    # pad_display() function
    grid = pad_display(parent(img1), parent(img2));
    offset = CartesianIndex(0, size(img1, 2));
    for m in matches
        draw!(grid, LineSegment(m[1], m[2] + offset))
    end
    grid
end
