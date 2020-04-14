
function timing_experiments(;method="HOFASM")
 #   kron_formation_times = []
    run_times = []
    n_values = [10,20,30,40,60,80,100]
    average_over = 100
    sigma = 1.0

    for n in n_values

#        kron_time = 0.0
        run_time = 0.0
        for i in 1:average_over+1
            if method == "HOFASM"
                _,  rt = synthetic_problem_new(n,sigma)
            elseif method == "HOM"
                _, rt, _ = synthetic_HOM(n,sigma)
            end

            if i == 0
                continue #skip due to compile time
            end
 #           kron_time += kt
            run_time += rt
        end

 #       push!(kron_formation_times,kron_time/average_over)
        push!(run_times,run_time/average_over)

        println("finished problem size $n")
    end

#    kron_formation_times,
    return  run_times
end

function accuracy_experiments(average_over::Int = 2;method="HOFASM")

    sigmas = [.025,.05,.0725,.1,.125,.15,.175,.2]

 #   kron_formation_times = []
    argsort_accuracies = []
    argsort_variances = []
    colsort_accuracies = []
    colsort_variances = []

    n = 30


    for sigma in sigmas

#        kron_time = 0.0
        test_argsort_accuracies = []
        test_colsort_accuracies = []

        for i in 1:average_over
            if method == "HOFASM"
                X , rt = synthetic_problem_new(n,sigma)
            elseif method == "HOM"
                X,_, p  = synthetic_HOM(n,sigma)
            end
            #transpose is need to maintain row major format
            assignment = build_assignment(X,use_colsort=true)
  #          argsort_accuracy = sum([1.0 for (i,j) in build_assignment(X) if i == j])/n
            if method != "HOM"
                colsort_accuracy = sum([1.0 for (i,j) in assignment if i == j])/n
            else
                colsort_accuracy = sum([1.0 for (i,j) in assignment if p[i] == j])/n
            end
  #          push!(test_argsort_accuracies,argsort_accuracy)
            push!(test_colsort_accuracies,colsort_accuracy)
        end



   #     push!(argsort_variances,var(test_argsort_accuracies))
   #     push!(argsort_accuracies,sum(test_argsort_accuracies)/average_over)

        push!(colsort_variances,var(test_colsort_accuracies))
        push!(colsort_accuracies,sum(test_colsort_accuracies)/average_over)
    end

#    kron_formation_times,
#    return argsort_variances, argsort_accuracies, colsort_variances, colsort_accuracies
    return colsort_variances, colsort_accuracies
end

function outlier_experiments(average_over::Int = 2)

    sigma = .1
    max_outliers = 20

 #   kron_formation_times = []
    argsort_accuracies = []
    argsort_variances = []
    colsort_accuracies = []
    colsort_variances = []

    n = 20


    for outliers = 0:max_outliers

#        kron_time = 0.0
        test_argsort_accuracies = []
        test_colsort_accuracies = []

        for i in 1:average_over
            X , _ = synthetic_problem_new(n,sigma,outliers)

            argsort_accuracy = sum([1.0 for (i,j) in build_assignment(X) if i == j])/n
            colsort_accuracy = sum([1.0 for (i,j) in build_assignment(X,use_colsort=true) if i == j])/n

            push!(test_argsort_accuracies,argsort_accuracy)
            push!(test_colsort_accuracies,colsort_accuracy)
        end



        push!(argsort_variances,var(test_argsort_accuracies))
        push!(argsort_accuracies,sum(test_argsort_accuracies)/average_over)

        push!(colsort_variances,var(test_colsort_accuracies))
        push!(colsort_accuracies,sum(test_colsort_accuracies)/average_over)
    end

#    kron_formation_times,
    return argsort_variances, argsort_accuracies, colsort_variances, colsort_accuracies
end

function scaling_experiments(average_over::Int = 2;display=false)

    sigma = .05
    outliers = 5
    scalings = range(1,stop=3,length=10)

 #   kron_formation_times = []
    argsort_accuracies = []
    argsort_variances = []
    colsort_accuracies = []
    colsort_variances = []

    n = 20


    for scale in scalings

#        kron_time = 0.0
        test_argsort_accuracies = []
        test_colsort_accuracies = []

        for i in 1:average_over
            X , _, _ ,source, target = synthetic_problem_new(n,sigma,outliers,scale)

            argsort_accuracy = sum([1.0 for (i,j) in build_assignment(X) if i == j])/n
            colsort_accuracy = sum([1.0 for (i,j) in build_assignment(X,use_colsort=true) if i == j])/n

            if display
                display_synthetic_result(source,target,n,build_assignment(X,use_colsort=true))
            end

            push!(test_argsort_accuracies,argsort_accuracy)
            push!(test_colsort_accuracies,colsort_accuracy)
        end



        push!(argsort_variances,var(test_argsort_accuracies))
        push!(argsort_accuracies,sum(test_argsort_accuracies)/average_over)

        push!(colsort_variances,var(test_colsort_accuracies))
        push!(colsort_accuracies,sum(test_colsort_accuracies)/average_over)
    end

#    kron_formation_times,
    return argsort_variances, argsort_accuracies, colsort_variances, colsort_accuracies
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