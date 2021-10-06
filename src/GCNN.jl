module GCNN

using Flux
using Flux: conv, convfilter, glorot_uniform, @functor
using SparseArrays

"""
Add an extra dimension to the end of an array.
"""
expand_dims(x) = reshape(x, size(x)..., 1)


"""
A conveniently bundled group of rotation matrices.
"""
struct RotGroup{RT<:AbstractSparseMatrix}
    Rs::Vector{RT}
end

@functor RotGroup


"""
Compute a sparse matrix that simultaneously does `nrots` interpolated rotations
of a m×n matrix. To compute the rotations, the image must be flattened, then
sparse matrix times dense vector multiply, then reshape the output vector into
a nrots×m×n array.
"""
function RotGroup(
        m::Integer, n::Integer, nrots::Integer, periodicity=2π, mask_disk::Bool=true)

    # TODO: until we test m != n case, I wouldn't trust it
    @assert m == n

    ci, cj = div(m, 2)+1, div(n, 2)+1
    c = min(ci, cj)

    Rs = Vector{SparseMatrixCSC{Float32, Int32}}(undef, nrots)

    for (r, θ) in enumerate(range(0, periodicity, length=nrots+1)[1:end-1])
        cosθ, sinθ = cos(θ), sin(θ)

        # COO sparse matrix values
        Is = Int32[]
        Js = Int32[]
        Vs = Float32[]

        for i in 1:m, j in 1:n
            if mask_disk && (i-ci)^2 + (j-cj)^2 > (c+0.5)^2
                continue
            end

            # flattened index
            l_input = (j-1)*m + i

            # compute fractional rotated indexes
            ir = cosθ * (i - ci) + sinθ * (j - cj) + ci
            jr = -sinθ * (i - ci) + cosθ * (j - cj) + cj

            # interpolate across 2x2 square
            ir1 = floor(Int, ir)
            ir2 = ir1 + 1
            jr1 = floor(Int, jr)
            jr2 = jr1 + 1

            wi = ir - ir1
            wj = jr - jr1

            for (irk, wik) in [(ir1, 1-wi), (ir2, wi)], (jrk, wjk) in [(jr1, 1-wj), (jr2, wj)]
                if 1 <= irk <= m && 1 <= jrk <= n
                    push!(Vs, wik * wjk)
                    l_output = (jrk-1)*m + irk
                    push!(Is, l_output)
                    push!(Js, l_input)
                end
            end

            Rs[r] = SparseMatrixCSC(sparse(Is, Js, Vs, m*n, m*n))
        end
    end

    return RotGroup(Rs)
end



"""
Apply each rotation matrix to an array w of shape [width, height, in_channels, out_channels]
produces a vector of rotated arrays each of the same shape.
"""
function (rg::RotGroup)(w::AbstractArray{T,4}) where {T}
    width, height, in_channels, out_channels = size(w)
    w_flat = reshape(w, (width*height, in_channels*out_channels))
    return [reshape(R*w_flat, (width, height, in_channels, out_channels)) for R in rg.Rs]
end


    
"""
Group convolutional layer lifting Z2 to SE2N with N orientations.
"""
struct RotGroupLiftingConv{F, WT<:AbstractArray, RT<:AbstractSparseMatrix}
    σ::F
    weight::WT
    rg::RotGroup{RT}
end

@functor RotGroupLiftingConv


function RotGroupLiftingConv(
        nrots::Integer, k::Integer, ch::Pair{<:Integer,<:Integer}, σ=identity;
        init=glorot_uniform, use_bias::Bool=true)

    weight = convfilter((k,k), ch; init)
    # bias = create_bias(weght, use_bias, size(weight, N))
    rg = RotGroup(k, k, nrots)

    return RotGroupLiftingConv(σ, weight, rg)
end


function (lyr::RotGroupLiftingConv)(x::AbstractArray)
    # Rotate the weights according to each rotation matrix, apply each matrix, then concatenate.
    return cat([expand_dims(lyr.σ.(conv(x, Rw))) for Rw in lyr.rg(lyr.weight)]..., dims=5)
end


struct RotGroupConv{F, WT<:AbstractArray, RT<:AbstractSparseArray}
    σ::F
    weight::WT
    Rs::Vector{RT}
end


function RotGroupConv()
    # TODO:
end





end # module
