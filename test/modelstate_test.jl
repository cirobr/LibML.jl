path = "./testpath/"
mkpath(path)


m1 = Flux.Dense(5,5)
s1 = Flux.state(m1)
p1 = Flux.params(m1)
fpfn = path * "modelname.jld2"
@test ml.saveModelState(fpfn, m1) === nothing


m2 = Flux.Dense(5,5)
p2 = Flux.params(m2)
@test p1 != p2   # parameters shall be different

ml.loadModelState!(fpfn, m2)
p2 = Flux.params(m2)
@test p1 == p2   # parameters shall be equal
rm(fpfn)


@test ml.saveModelStateCB(path, m1) === nothing


rm(path, recursive=true, force=true)
