module LibML


# libraries
export IoU, IoU_loss
export saveModelState, loadModelState
export trainModel!, testModel

import Flux
import Statistics: mean
import BSON

include("./modelstate.jl")               # saveModelState, loadModelState
include("./lossfunctions.jl")            # IoU
include("./training.jl")                 # trainModel!


end   # module