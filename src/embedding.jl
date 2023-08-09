using LinearAlgebra
using CUDA
using Flux
"""
Module to encode the embedding of the atomic species.
"""


function get_input_layer(coords, atomic_species, n_species)
    # Prepare the embedding for the atomic species
    
    embedding = Embedding(n_species => 4)

end


@doc raw"""
    build_network(n_species, n_embedding, n_cutoff)

Builds the network that will be used to predict the energy of the system.
"""
function build_network(n_species, n_embedding, n_cutoff)
    embedding = Embedding(n_species => n_embedding)
    DNN = Chain(
                Dense(n_embedding + 2*n_cutoff, 32, relu),
                Dense(32, 16, relu),
                Dense(16, 1)
                )

    function network(coords, atomic_species)
        # Prepare the embedding for the atomic species

        # Prepare the descriptors
        n_atoms = size(coords, 1)
        descriptors = zeros(n_cutoff, n_atoms, 2)
        atomic_environment = zeros(n_atoms)

        distances = zeros(n_atoms)
        atomic_layers = []

        output = 0.0
        for i in 1:n_atoms
            distances .= 0
            for j in i+1:n_atoms
                distances[j] = norm(coords[i, :] - coords[j, :])
            end
            atomic_environment = embedding(atomic_species)
            atomic_environment = sortperm(distances)
            sorted_distances = sort(distances)

            sort_dist = 1.0 ./ sorted_distances[1:n_cutoff]
            atoms =  atomic_environment[1:n_cutoff]

            # Concatenate the embedding with the coordinates
            input_layer = Flux.cat(embedding(atomic_species[i]), sort_dist, atoms, dims=1)
            energy += DNN(input_layer)
        end

        return output
    end

    return network
end


function lennard_jones(r, sigma, epsilon)
    return 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)
end

function potential(coords, species, sigma_dict, epsilon_dict)
    # Compute the potential energy of the system
    n_atoms = size(coords, 1)
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
    coords = randn(dimensions, n_random)
    species = rand(1:n_species, n_random)

    sigma_dict = Dict(1 => 1.0, 2 => 2.0)
    epsilon_dict = Dict(1 => 1.0, 2 => 2.0)

    energies = zeros(n_random)
    forces = zeros(dimensions*n_atoms, n_random)

    for i in 1:n_random
        energies[i] = potential(coords[:, i], species, sigma_dict, epsilon_dict)
    end






