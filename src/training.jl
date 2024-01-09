function trainModel!(model, data, optstate, lossfn; epochs::Int=1, verbose=false)
    len_data    = length(data)
    epochLosses = Vector{Float32}(undef, epochs)              # buffer for epoch losses
    stepLosses  = Vector{Float32}(undef, epochs * len_data)   # buffer for step losses

    for epoch in 1:epochs
        vectorsteplosses = Vector{Float32}(undef, len_data)   # buffer for single epoch loss calculation

        for (i, (X,y)) in enumerate(data)
            loss, grads = Flux.withgradient(model) do m
                yhat = m(X)
                lossfn(yhat, y)
            end

            Flux.update!(optstate, model, grads[1])

            vectorsteplosses[i]                = loss         # step loss for epoch loss calculation
            stepLosses[(epoch-1)*len_data + i] = loss         # store step loss history
        end

        epochLosses[epoch] = mean(vectorsteplosses)           # epoch loss calculation
    end

    if verbose   return epochLosses, stepLosses
    else         return
    end
end


# lossfns is a vector of loss functions
function testModel(model, data, lossfns)
    losses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    for (i, (X,y)) in enumerate(data)
        yhat = model(X)
        for (j, lossfn) in enumerate(lossfns)
            losses[i,j] = lossfn(yhat, y)
        end
    end

    return losses
end