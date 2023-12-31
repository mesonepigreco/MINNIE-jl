using LinearAlgebra
using CUDA
using Flux
using Zygote: @ignore
using Flux.Optimise: update!, Descent
using Flux: params
using JLD2

using ChainRulesCore
using ChainRulesCore: rrule

"""
Module to encode the embedding of the atomic species. """


# Create the neural network
const n_species = 2
const n_embedding = 2
const n_atoms = 2
const n_dims = 1
const n_cutoff = 2
const n_rand = 20000
const n_test_set = 1000

struct CustomModel
    embedding
    DNN
end

function get_dist(coords :: Matrix{T}) where T
    n_rand = size(coords, 1)
    distance = zeros(T, 1, n_rand)
    distance[1, :] = abs.(coords[:, 1] .- coords[:, 2])
    return distance
end

function ChainRulesCore.rrule(::typeof(get_dist), coords :: Matrix{T}) where T
    function pullback(∇dist)
        n_rand = size(coords, 1)
        ∇coords = zeros(T, size(coords))
        for i in 1:n_rand
            if coords[i, 1] > coords[i, 2]
                ∇coords[i, 1] = ∇dist[1, i]
                ∇coords[i, 2] = -∇dist[1, i]
            else
                ∇coords[i, 1] = -∇dist[1, i]
                ∇coords[i, 2] = ∇dist[1, i]
            end
        end 

        return NoTangent(), ∇coords
    end 
    return get_dist(coords), pullback
end

function (m::CustomModel)(coords, atomic_species)
    n_rand = size(coords)[1]
    distance = get_dist(coords)
    
    # Concatenate the embedding with the coordinates
    emblayer = m.embedding(permutedims(atomic_species, (2,1)))
    emblayer_2 = reshape(permutedims(emblayer, (3, 1, 2)), n_rand, n_atoms * n_embedding)
    input_layer = Flux.cat(emblayer_2', distance, dims=1)
    output = m.DNN(input_layer)

    return output
end

Flux.@functor CustomModel


function lennard_jones(r, sigma, epsilon)
    r = sigma * √( (r/sigma)^2 + 0.4 )
    return 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)
end

function potential(coords, species, sigma_dict, epsilon_dict)
    # Compute the potential energy of the system
    coords = reshape(coords, n_atoms, n_dims)
    energy = 0.0
    for i in 1:n_atoms
        for j in i+1:n_atoms
            r = norm(coords[i, :] - coords[j, :])
            sigma = (sigma_dict[species[i]] + sigma_dict[species[j]]) / 2
            epsilon = sqrt(epsilon_dict[species[i]] * epsilon_dict[species[j]])
            energy += lennard_jones(r, sigma, epsilon)
        end
    end
    return energy
end


function generate_train_set(n_random, n_species, n_atoms, dimensions)
    coords = randn(Float32, n_random, n_atoms * dimensions)
    species = rand(1:n_species, n_random, n_atoms)

    sigma_dict = Dict(1 => Float32(0.5), 2 => Float32(0.3))

    epsilon_dict = Dict(1 => Float32(0.2), 2 => Float32(.4))

    energies = zeros(Float32, n_random)

    for i in 1:n_random
        energies[i] = potential(coords[i, :], species[i, :], sigma_dict, epsilon_dict)
    end

    return coords, species, energies
end

function loss(model, coords, species, energies)
    pred_energy = model(coords, species)
    return Flux.mse(pred_energy, reshape(energies, (1, length(energies))))
end

function main()
    embedding = Embedding(n_species => n_embedding)
    DNN = Chain(
                Dense(n_embedding*n_atoms + 1, 32, relu),
                Dense(32, 16, relu),
                Dense(16, 1)
                )

    model = CustomModel(embedding, DNN)

    coords, species, energies = generate_train_set(n_rand, n_species, n_atoms, n_dims)
    test_coords, test_species, test_energies = generate_train_set(n_test_set, n_species, n_atoms, n_dims)



    # Compare the gradient with finite differences
    old_loss = loss(model, test_coords, test_species, test_energies)
    min_loss = old_loss
    η :: Float32 = 0.001
    for iter in 1:1000
        # Pick the subset of configurations 
        mini_batch = rand(1:size(coords, 1), 100)

        
        batch_coords = coords[mini_batch, :]
        batch_species = species[mini_batch, :]
        batch_energies = energies[mini_batch]

        grads = gradient(params(model)) do
            loss(model, batch_coords, batch_species, batch_energies)
        end

        
        for p in params(model)
            update!(p, η * grads[p])
        end


        # Compute the loss of the training set
        train_loss = loss(model, test_coords, test_species, test_energies)

        if (train_loss > old_loss)
            η = η / 1.2
        end

        if (train_loss < min_loss)
            # Save the model
            jldsave("model-checkpoint.jld2"; model_state=Flux.state(model))
            min_loss = train_loss
        end
        old_loss = train_loss

        println("$iter    $(train_loss)")
    end
end


main()

