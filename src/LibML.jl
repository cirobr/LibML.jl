module LibML


# libraries
using Flux
using BSON

# include("./trainepoch-flux-v013.jl")     # deprecated
# include("./trainepoch-flux-v014.jl")     # disabled, as it uses CUDA
include("./modelstate.jl")               # loadModelState, saveModelState
include("./lossfunctions.jl")            # IoU


end   # module