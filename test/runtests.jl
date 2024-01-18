using LibML
using Flux
using Random
using Test

@testset "LibML.jl" begin
    include("./lossfunctions_test.jl")
    include("./modelstate_test.jl")
end
