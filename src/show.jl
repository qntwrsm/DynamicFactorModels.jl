#=
show.jl

    Provides a collection of methods for pretty printing dynamic factor models 
    and its components.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/11/01
=#

# union type for all models
Models = Union{DynamicFactorModel, AbstractFactorProcess, AbstractMeanSpecification, AbstractErrorModel}

function show(io::IO, model::Models)
    print(io, nameof(typeof(model)))
    println(io, "(")
    for p in fieldnames(typeof(model))
        print(io, p)
        print(io, ": ")
        println(io, getfield(model, p))
    end
    print(io, ")")
end