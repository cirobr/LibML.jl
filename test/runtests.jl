using LibML
using Random
using Test

@testset "LibML.jl" begin
    Random.seed!(1)
    yhat = rand(Bool, 5,5)
    y    = rand(Bool, 5,5)
   
    @test LibML.IoU(yhat, y) == 0.42105263f0 || error("IoU failed")
    @test LibML.IoU_loss(yhat, y) == 0.57894737f0 || error("IoU_loss failed")
end
