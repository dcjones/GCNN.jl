module GCNN

export RotGroupConv, RotGroupConvTranspose

using Flux
using Flux: conv, ∇conv_data, convfilter, glorot_uniform, @functor, DenseConvDims



"""
Add an extra dimension to the end of an array.
"""
expand_dims(x) = reshape(x, size(x)..., 1)


"""
A conveniently bundled group of rotation matrices.
"""
struct RotGroup{RT<:AbstractMatrix}
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

    Rs = Vector{Matrix{Float32}}(undef, nrots)

    for (r, θ) in enumerate(range(0, periodicity, length=nrots+1)[1:end-1])
        cosθ, sinθ = cos(θ), sin(θ)

        # COO sparse matrix values
        R = zeros(Float32, (m*n, m*n))

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
                    l_output = (jrk-1)*m + irk
                    R[l_output, l_input] = wik * wjk
                end
            end

            Rs[r] = R
        end
    end

    # TODO: It's probably better to use dense matrices since these will be
    # small. (E.g. to rotate a 5x5 rotation kernel, we only need a 25x25 matrix).

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


Base.length(rg::RotGroup) = length(rg.Rs)


    
"""
Group convolutional layer.
"""
struct RotGroupConv{F, WT<:AbstractArray, BT, RT<:AbstractMatrix, M, N}
    rg::RotGroup{RT}
    weight::WT
    bias::BT
    σ::F
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end

@functor RotGroupConv


"""
    RotGroupConv(nrots, filter, in => out, σ = identity)

Construct a Group CNN (G-CNN) layer with `nrots` equally spaced rotations, and
a `(filter, filter)` convolution kernel, with `in` input channels and `out`
output channels, with an optional elementwise nonlinearity `σ`.
"""
function RotGroupConv(
        nrots::Integer, filter::Integer, ch::Pair{<:Integer,<:Integer}, σ=identity;
        init=glorot_uniform, stride::Int=1, pad=0, dilation=1, use_bias::Bool=true)

    weight = convfilter((filter, filter), ch; init)
    # bias = create_bias(weght, use_bias, size(weight, N))
    rg = RotGroup(filter, filter, nrots)

    return RotGroupConv(rg, weight, σ, stride, pad, dilation, use_bias)
end


function RotGroupConv(
        rg::RotGroup, weight::AbstractArray{T, N}, σ, stride, pad, dilation, use_bias) where {T, N}

    stride = Flux.expand(Val(N-2), stride)
    dilation = Flux.expand(Val(N-2), dilation)
    pad = Flux.calc_padding(Conv, pad, size(weight)[1:N-2], dilation, stride)
    bias = Flux.create_bias(weight, use_bias, size(weight, N))
    return RotGroupConv(rg, weight, bias, σ, stride, pad, dilation)
end


"""
Rotation group convolution applied to an array with shape [width, height, in_channels, batches] and
producing an array of shape [width, height, out_channels, batches, nrotations].
"""
function (lyr::RotGroupConv)(x::AbstractArray{T,4}) where {T}
    # Rotate the weights according to each rotation matrix, apply each matrix, then concatenate.
    bias = reshape(lyr.bias, ntuple(_ -> 1, length(lyr.stride))..., :, 1)
    cdims = DenseConvDims(x, lyr.weight; stride=lyr.stride, padding=lyr.pad, dilation=lyr.dilation)
    return cat([expand_dims(lyr.σ.(conv(x, Rw, cdims) .+ bias)) for Rw in lyr.rg(lyr.weight)]..., dims=5)
end


"""
Rotation group convolution applied to an array with shape `[width, height,
in_channels, batches, nrotations]` and producing an array of shape `[width,
height, out_channels, batches, nrotations]`.
"""
function (lyr::RotGroupConv)(x::AbstractArray{T,5}) where {T}
    # TODO:
    nrots = length(lyr.rg)
    @assert size(x, 5) == nrots

    bias = reshape(lyr.bias, ntuple(_ -> 1, length(lyr.stride))..., :, 1)
    cdims = DenseConvDims(size(x)[1:end-1], size(lyr.weight); stride=lyr.stride, padding=lyr.pad, dilation=lyr.dilation)

    Rws = lyr.rg(lyr.weight)
    Rxs = [view(x, :, :, :, :, i) for i in 1:nrots]

    # gradients don't work with zip
    #return cat([expand_dims(lyr.σ.(conv(Rx, Rw))) for (Rx, Rw) in zip(Rxs, Rws)]..., dims=5)

    return cat([expand_dims(lyr.σ.(conv(Rxs[k], Rws[k], cdims) .+ bias)) for k in 1:nrots]..., dims=5)
end


"""
Group transposed convolutional layer.
"""
struct RotGroupConvTranspose{F, WT<:AbstractArray, BT, RT<:AbstractMatrix, M, N}
    rg::RotGroup{RT}
    weight::WT
    bias::BT
    σ::F
    # TODO: Make this a tuple so everythig works.
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end

@functor RotGroupConvTranspose


"""
    RotGroupConvTranspose(nrots, filter, in => out, σ = identity)

Construct a Group CNN (G-CNN) transposed convolution layer with `nrots` equally
spaced rotations, and a `(filter, filter)` convolution kernel, with `in` input
channels and `out` output channels, with an optional elementwise nonlinearity
`σ`.
"""
function RotGroupConvTranspose(
        nrots::Integer, filter::Integer, ch::Pair{<:Integer,<:Integer}, σ=identity;
        init=glorot_uniform, stride::Int=1, pad=0, dilation=1, use_bias::Bool=true)

    weight = convfilter((filter, filter), reverse(ch); init)

    # bias = create_bias(weght, use_bias, size(weight, N))
    rg = RotGroup(filter, filter, nrots)

    return RotGroupConvTranspose(rg, weight, σ, stride, pad, dilation, use_bias)
end


function RotGroupConvTranspose(
        rg::RotGroup, weight::AbstractArray{T, N}, σ, stride, pad, dilation, use_bias) where {T, N}
    stride = Flux.expand(Val(N-2), stride)
    dilation = Flux.expand(Val(N-2), dilation)
    pad = Flux.calc_padding(ConvTranspose, pad, size(weight)[1:N-2], dilation, stride)
    bias = Flux.create_bias(weight, use_bias, size(weight, N-1))
    return RotGroupConvTranspose(rg, weight, bias, σ, stride, pad, dilation)
end


function conv_transpose_dims(c::RotGroupConvTranspose, xsize::NTuple{N, Int}) where {N}
    # Calculate size of "input", from ∇conv_data()'s perspective...
    combined_pad = (c.pad[1:2:end] .+ c.pad[2:2:end])
    I = (xsize[1:end-2] .- 1).*c.stride .+ 1 .+ (size(c.weight)[1:end-2] .- 1).*c.dilation .- combined_pad
    C_in = size(c.weight)[end-1]
    batch_size = xsize[end]
    # Create DenseConvDims() that looks like the corresponding conv()
    return DenseConvDims((I..., C_in, batch_size), size(c.weight);
                        stride=c.stride,
                        padding=c.pad,
                        dilation=c.dilation,
    )
end


"""
Rotation group transposed convolution applied to an array with shape [width,
height, in_channels, batches] and producing an array of shape [width, height,
out_channels, batches, nrotations].
"""
function (lyr::RotGroupConvTranspose)(x::AbstractArray{T,4}) where {T}
    # Rotate the weights according to each rotation matrix, apply each matrix, then concatenate.
    cdims = conv_transpose_dims(lyr, size(x))
    bias = reshape(lyr.bias, map(_->1, lyr.stride)..., :, 1)

    # return cat([expand_dims(lyr.σ.(∇conv_data(x, Rw, cdims) .+ lyr.bias)) for Rw in lyr.rg(lyr.weight)]..., dims=5)
    return cat([expand_dims(lyr.σ.(∇conv_data(x, Rw, cdims) .+ bias)) for Rw in lyr.rg(lyr.weight)]..., dims=5)
end


"""
Rotation group transposed convolution applied to an array with shape `[width,
height, in_channels, batches, nrotations]` and producing an array of shape
`[width, height, out_channels, batches, nrotations]`.
"""
function (lyr::RotGroupConvTranspose)(x::AbstractArray{T,5}) where {T}
    nrots = length(lyr.rg)
    @assert size(x, 5) == nrots

    Rws = lyr.rg(lyr.weight)
    Rxs = [view(x, :, :, :, :, i) for i in 1:nrots]
    cdims = conv_transpose_dims(lyr, size(x)[1:end-1])
    bias = reshape(lyr.bias, map(_->1, lyr.stride)..., :, 1)

    # gradients don't work with zip
    #return cat([expand_dims(lyr.σ.(conv(Rx, Rw))) for (Rx, Rw) in zip(Rxs, Rws)]..., dims=5)

    return cat([expand_dims(lyr.σ.(∇conv_data(Rxs[k], Rws[k], cdims) .+ bias)) for k in 1:nrots]..., dims=5)
end


end # module
