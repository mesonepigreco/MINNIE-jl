module Minnie

const G2_LEN  = 2
const G4_LEN  = 4

struct SymmetriFunction{T}
    n_g2 :: Int
    n_g4 :: Int
    vector_g2 :: Matrix{T}
    vector_g4 :: Matrix{T}
end

end # module Minnie
