model = Flux.Conv((3,3), 3=>1, relu)
@test ml.countParams(model) == 28