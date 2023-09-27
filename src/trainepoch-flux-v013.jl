# Flux v0.13 and before


# data entirely on gpu
function trainModel1!(loss, ps, data, opt, nearZero, lossVector)
    
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    # ps = Params(ps)   # Zygote.Params(ps) ???
    cb = Flux.throttle( (() -> @show training_loss), 10)
    es = Flux.plateau(() -> training_loss, 3; min_dist = nearZero)

    for d in data
        gs = Flux.gradient(ps) do   
            training_loss = loss(d...)   # strategy to obtain losses
            return training_loss
        end
        
        # update weights
        Flux.update!(opt, ps, gs)
        
        # losses
        push!(lossVector, training_loss)

        # callbacks
        cb()

        # early stop
        if es()   break   end
    end

    return es()
end


# iterate over minibatches on cpu and load to gpu
function trainModel2!(loss, ps, data, opt, nearZero, lossVector)
    
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    # ps = Params(ps)   # Zygote.Params(ps) ???
    cb = Flux.throttle( (() -> @show training_loss), 10)
    es = Flux.plateau(() -> training_loss, 3; min_dist = nearZero)

    for (X, y) in data
        gs = Flux.gradient(ps) do   
            training_loss = loss(Flux.gpu(X), Flux.gpu(y))   # strategy to obtain losses
            return training_loss
        end

        # update weights
        Flux.update!(opt, ps, gs)
        
        # losses
        push!(lossVector, training_loss)

        # callbacks
        cb()

        # early stop
        if es()   break   end
    end

    return es()
end


# CuIterator
function trainModel3!(loss, ps, data, opt, nearzero, lossVector)
    
    # training_loss is declared local so it will be available for logging outside the gradient calculation.
    local training_loss

    # ps = Params(ps)   # Zygote.Params(ps) ???
    # cb = Flux.throttle( (() -> @show training_loss), 30) #; leading=false, trailing=false)
    es1 = Flux.plateau(() -> training_loss,          10; min_dist = nearzero)
    es2 = Flux.early_stopping(() -> training_loss,   10; min_dist = nearzero, init_score = Inf)

    for (X, y) in CuIterator(data)
        gs = Flux.gradient(ps) do   
            training_loss = loss(X, y)   # strategy to obtain losses
            return training_loss
        end

        # update weights
        Flux.update!(opt, ps, gs)
        
        # losses
        push!(lossVector, training_loss)

        # callbacks
        # cb()

        # early stop
        if es1() | es2()   break   end
    end

    return
end
