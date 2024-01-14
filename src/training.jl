# epochLosses = Vector{Float32}()
# stepLosses  = Vector{Float32}()
function trainModel!(model, data, optstate, lossfn,
                     epochLosses::Vector{Float32}, stepLosses::Vector{Float32};
                     epochs::Int=1)
                     
    len_data    = length(data)
    epochLosses = Vector{Float32}(undef, epochs)              # buffer for epoch losses
    stepLosses  = Vector{Float32}(undef, epochs * len_data)   # buffer for step losses

    for epoch in 1:epochs
        vectorsteplosses = Vector{Float32}(undef, len_data)   # buffer for single epoch loss calculation

        for (i, (X,y)) in enumerate(data)
            loss, grads = Flux.withgradient(model) do m
                lossfn(m(X), y)
            end

            Flux.update!(optstate, model, grads[1])

            vectorsteplosses[i]                = loss         # step loss for epoch loss calculation
            stepLosses[(epoch-1)*len_data + i] = loss         # store step loss history
        end

        epochLosses[epoch] = mean(vectorsteplosses)           # epoch loss calculation
    end

    return
end


# stepLosses  = Vector{Float32}()
function testModel(model, data, lossfns::Vector{Any},
                   stepLosses::Vector{Float32})

    stepLosses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    for (i, (X,y)) in enumerate(data)
        yhat = model(X)
        for (j, lossfn) in enumerate(lossfns)
            stepLosses[i,j] = lossfn(yhat, y)
        end
    end

    return
end