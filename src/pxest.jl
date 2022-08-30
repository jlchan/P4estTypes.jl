const Locidx = P4est.p4est_locidx_t
const Gloidx = P4est.p4est_gloidx_t

struct Quadrant{X,T,P}
    pointer::P
end

level(quadrant::Quadrant) = quadrant.pointer.level
storeuserdata!(quadrant::Quadrant{X,T}, data::T) where {X,T} =
    unsafe_store!(Ptr{T}(quadrant.pointer.p.user_data), data)
loaduserdata(quadrant::Quadrant{X,T}) where {X,T} =
    unsafe_load(Ptr{T}(quadrant.pointer.p.user_data))

struct Tree{X,T,P} <: AbstractArray{Quadrant,1}
    pointer::P
end

Base.size(t::Tree) = (t.pointer.quadrants.elem_count,)
function Base.getindex(t::Tree{X,T}, i::Int) where {X,T}
    @boundscheck checkbounds(t, i)
    GC.@preserve t begin
        Q = X == 4 ? p4est_quadrant : p8est_quadrant
        quadrant = Ptr{Q}(t.pointer.quadrants.array + sizeof(Q) * (i - 1))
        return Quadrant{X,T,Ptr{Q}}(quadrant)
    end
end
Base.IndexStyle(::Tree) = IndexLinear()

offset(tree::Tree) = tree.pointer.quadrants_offset

mutable struct Pxest{X,T,P,C} <: AbstractArray{Tree,1}
    pointer::P
    connectivity::C
    comm::MPI.Comm
    function Pxest{4}(
        pointer::Ptr{P4est.LibP4est.p4est},
        connectivity::Connectivity{4},
        comm::MPI.Comm,
        ::Type{T},
    ) where {T}
        forest = new{4,T,typeof(pointer),typeof(connectivity)}(pointer, connectivity, comm)
        finalizer(forest) do p
            p4est_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
    function Pxest{8}(
        pointer::Ptr{P4est.LibP4est.p8est},
        connectivity::Connectivity{8},
        comm::MPI.Comm,
        ::Type{T},
    ) where {T}
        forest = new{8,T,typeof(pointer),typeof(connectivity)}(pointer, connectivity, comm)
        finalizer(forest) do p
            p8est_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
end

function pxest(
    connectivity::Connectivity{4};
    comm = MPI.COMM_WORLD,
    min_quadrants = 0,
    min_level = 0,
    fill_uniform = true,
    data_type = Nothing,
    init = nothing,
    user_pointer = C_NULL,
)
    MPI.Initialized() || MPI.Init()

    init_callback = isnothing(init) ? C_NULL : _init_callback_generate(forest, init)

    pointer = p4est_new_ext(
        comm,
        connectivity,
        min_quadrants,
        min_level,
        fill_uniform,
        sizeof(data_type),
        init_callback,
        user_pointer,
    )
    return Pxest{4}(pointer, connectivity, comm, data_type)
end

function pxest(
    connectivity::Connectivity{8};
    comm = MPI.COMM_WORLD,
    min_quadrants = 0,
    min_level = 0,
    fill_uniform = true,
    data_type = Nothing,
    init = nothing,
    user_pointer = C_NULL,
)
    MPI.Initialized() || MPI.Init()

    init_callback = isnothing(init) ? C_NULL : _init_callback_generate(forest, init)

    pointer = p8est_new_ext(
        comm,
        connectivity,
        min_quadrants,
        min_level,
        fill_uniform,
        sizeof(data_type),
        init_callback,
        user_pointer,
    )
    return Pxest{8}(pointer, connectivity, comm, data_type)
end

mutable struct GhostLayer{X,P}
    pointer::P
    function GhostLayer{4}(pointer::Ptr{P4est.LibP4est.p4est_ghost_t})
        ghost = new{4,typeof(pointer)}(pointer)
        finalizer(ghost) do p
            p4est_ghost_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
    function GhostLayer{8}(pointer::Ptr{P4est.LibP4est.p8est_ghost_t})
        ghost = new{8,typeof(pointer)}(pointer)
        finalizer(ghost) do p
            p8est_ghost_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
end

function ghostlayer(forest::Pxest{4}; connection = CONNECT_FULL(Val(4)))
    return GhostLayer{4}(p4est_ghost_new(forest, connection))
end

function ghostlayer(forest::Pxest{8}; connection = CONNECT_FULL(Val(8)))
    return GhostLayer{8}(p8est_ghost_new(forest, connection))
end

function Base.unsafe_convert(
    ::Type{Ptr{p4est_ghost_t}},
    p::GhostLayer{4,Ptr{p4est_ghost_t}},
)
    return p.pointer
end

function Base.unsafe_convert(
    ::Type{Ptr{p8est_ghost_t}},
    p::GhostLayer{8,Ptr{p8est_ghost_t}},
)
    return p.pointer
end

mutable struct LNodes{X,P}
    pointer::P
    function LNodes{4}(pointer::Ptr{P4est.LibP4est.p4est_lnodes})
        nodes = new{4,typeof(pointer)}(pointer)
        finalizer(nodes) do p
            p4est_lnodes_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
    function LNodes{8}(pointer::Ptr{P4est.LibP4est.p8est_lnodes})
        nodes = new{8,typeof(pointer)}(pointer)
        finalizer(nodes) do p
            p8est_lnodes_destroy(p.pointer)
            p.pointer = C_NULL
            return
        end
    end
end

function lnodes(forest::Pxest{4}; ghost = nothing, degree = 1)
    if isnothing(ghost)
        ghost = ghostlayer(forest)
    end
    return LNodes{4}(p4est_lnodes_new(forest, ghost, degree))
end

function lnodes(forest::Pxest{8}; ghost = nothing, degree = 1)
    if isnothing(ghost)
        ghost = ghostlayer(forest)
    end
    return LNodes{8}(p8est_lnodes_new(forest, ghost, degree))
end

function Base.unsafe_convert(::Type{Ptr{p4est_lnodes}}, p::LNodes{4,Ptr{p4est_lnodes}})
    return p.pointer
end

function Base.unsafe_convert(::Type{Ptr{p8est_lnodes}}, p::LNodes{8,Ptr{p8est_lnodes}})
    return p.pointer
end

typeofquadrantuserdata(::Pxest{X,T}) where {X,T} = T

lengthoflocalquadrants(p::Pxest) = p.pointer.local_num_quadrants
lengthofglobalquadrants(p::Pxest) = p.pointer.global_num_quadrants

function Base.unsafe_convert(::Type{Ptr{p4est}}, p::Pxest{4,T,Ptr{p4est}}) where {T}
    return p.pointer
end
function Base.unsafe_convert(::Type{Ptr{p8est}}, p::Pxest{8,T,Ptr{p8est}}) where {T}
    return p.pointer
end

Base.size(p::Pxest) = (p.pointer.trees.elem_count,)
function Base.getindex(p::Pxest{X,T}, i::Int) where {X,T}
    @boundscheck checkbounds(p, i)
    GC.@preserve p begin
        tree = Ptr{p4est_tree}(p.pointer.trees.array + sizeof(p4est_tree) * (i - 1))
        return Tree{X,T,Ptr{p4est_tree}}(tree)
    end
end
Base.IndexStyle(::Pxest) = IndexLinear()

function _p4est_volume_callback_generate(forest, ghost, user_data, volume_callback)
    Ccallback, _ =
        Cfunction{Cvoid,Tuple{Ptr{p4est_iter_volume_info_t},Ptr{Cvoid}}}() do info, _
            T = typeofquadrantuserdata(forest)
            quadrant = Quadrant{4,T,Ptr{p4est_quadrant}}(info.quad)
            volume_callback(
                forest,
                ghost,
                quadrant,
                info.quadid + 1,
                info.treeid + 1,
                user_data,
            )
            return
        end

    return Ccallback
end

function iterateforest(
    forest::Pxest{4};
    user_data = nothing,
    ghost_layer = nothing,
    volume_callback = nothing,
    face_callback = nothing,
    corner_callback = nothing,
)

    _volume_callback =
        isnothing(volume_callback) ? C_NULL :
        _p4est_volume_callback_generate(forest, ghost_layer, user_data, volume_callback)
    @assert face_callback === nothing
    @assert corner_callback === nothing

    p4est_iterate(forest, C_NULL, C_NULL, _volume_callback, C_NULL, C_NULL)

    return
end

function _coarsen_callback_generate(forest::Pxest{4,T}, coarsen) where {T}
    Ccallback, _ = Cfunction{
        Cint,
        Tuple{Ptr{p4est},p4est_topidx_t,Ptr{Ptr{p4est_quadrant}}},
    }() do _, which_tree, children_ptr
        children_ptr = unsafe_wrap(Array, children_ptr, 4)
        children = Quadrant{4,T,Ptr{p4est_quadrant}}.(children_ptr)
        return coarsen(forest, which_tree + 1, children) ? one(Cint) : zero(Cint)
    end

    return Ccallback
end

function _refine_callback_generate(forest::Pxest{4,T}, refine) where {T}
    Ccallback, _ = Cfunction{
        Cint,
        Tuple{Ptr{p4est},p4est_topidx_t,Ptr{p4est_quadrant}},
    }() do _, which_tree, quadrant
        quadrant = Quadrant{4,T,Ptr{p4est_quadrant}}(quadrant)
        return refine(forest, which_tree + 1, quadrant) ? one(Cint) : zero(Cint)
    end

    return Ccallback
end

function _init_callback_generate(forest::Pxest{4,T}, init) where {T}
    Ccallback, _ = Cfunction{
        Cvoid,
        Tuple{Ptr{p4est},p4est_topidx_t,Ptr{p4est_quadrant}},
    }() do _, which_tree, quadrant
        quadrant = Quadrant{4,T,Ptr{p4est_quadrant}}(quadrant)
        init(forest, which_tree + 1, quadrant)
        return
    end

    return Ccallback
end

function _replace_callback_generate(forest::Pxest{4,T}, replace) where {T}
    Ccallback, _ = Cfunction{
        Cvoid,
        Tuple{
            Ptr{p4est},
            p4est_topidx_t,
            Cint,
            Ptr{Ptr{p4est_quadrant_t}},
            Cint,
            Ptr{Ptr{p4est_quadrant_t}},
        },
    }() do _, which_tree, num_outgoing, outgoing_ptr, num_incoming, incoming_ptr
        @show num_outgoing
        @show num_incoming
        outgoing_ptr = unsafe_wrap(Array, outgoing_ptr, num_outgoing)
        incoming_ptr = unsafe_wrap(Array, incoming_ptr, num_incoming)
        outgoing = Quadrant{4,T,Ptr{p4est_quadrant}}.(outgoing_ptr)
        incoming = Quadrant{4,T,Ptr{p4est_quadrant}}.(incoming_ptr)
        replace(forest, which_tree + 1, outgoing, incoming)
        return
    end

    return Ccallback
end

function _weight_callback_generate(forest::Pxest{4,T}, weight) where {T}
    Ccallback, _ = Cfunction{
        Cint,
        Tuple{Ptr{p4est},p4est_topidx_t,Ptr{p4est_quadrant}},
    }() do _, which_tree, quadrant
        quadrant = Quadrant{4,T,Ptr{p4est_quadrant}}(quadrant)
        return weight(forest, which_tree + 1, quadrant)
    end

    return Ccallback
end

@inline function _coarsen_ext(
    forest::Pxest{4},
    recursive,
    callback_orphans,
    coarsen,
    init,
    replace,
)
    p4est_coarsen_ext(forest, recursive, callback_orphans, coarsen, init, replace)
end

@inline function _coarsen_ext(
    forest::Pxest{8},
    recursive,
    callback_orphans,
    coarsen,
    init,
    replace,
)
    p8est_coarsen_ext(forest, recursive, callback_orphans, coarsen, init, replace)
end

function coarsen!(
    forest::Pxest{X};
    recursive = false,
    callback_orphans = false,
    coarsen = nothing,
    init = nothing,
    replace = nothing,
) where {X}

    GC.@preserve forest begin
        coarsen_callback =
            isnothing(coarsen) ? C_NULL : _coarsen_callback_generate(forest, coarsen)
        init_callback = isnothing(init) ? C_NULL : _init_callback_generate(forest, init)
        replace_callback =
            isnothing(replace) ? C_NULL : _replace_callback_generate(forest, replace)

        _coarsen_ext(
            forest,
            recursive,
            callback_orphans,
            coarsen_callback,
            init_callback,
            replace_callback,
        )
    end

    return
end

@inline function _refine_ext(forest::Pxest{4}, recursive, maxlevel, refine, init, replace)
    p4est_refine_ext(forest, recursive, maxlevel, refine, init, replace)
end

@inline function _refine_ext(forest::Pxest{8}, recursive, maxlevel, refine, init, replace)
    p8est_refine_ext(forest, recursive, maxlevel, refine, init, replace)
end

function refine!(
    forest::Pxest{X};
    recursive = false,
    maxlevel = -1,
    refine = nothing,
    init = nothing,
    replace = nothing,
) where {X}

    refine_callback = isnothing(refine) ? C_NULL : _refine_callback_generate(forest, refine)
    init_callback = isnothing(init) ? C_NULL : _init_callback_generate(forest, init)
    replace_callback =
        isnothing(replace) ? C_NULL : _replace_callback_generate(forest, replace)

    _refine_ext(
        forest,
        recursive,
        maxlevel,
        refine_callback,
        init_callback,
        replace_callback,
    )

    return
end

@inline function _balance_ext(forest::Pxest{4}, connect, init, replace)
    p4est_balance_ext(forest, connect, init, replace)
end

@inline function _balance_ext(forest::Pxest{8}, connect, init, replace)
    p8est_balance_ext(forest, connect, init, replace)
end

function balance!(
    forest::Pxest{X};
    connect = CONNECT_FULL(Val(X)),
    init = nothing,
    replace = nothing,
) where {X}

    init_callback = isnothing(init) ? C_NULL : _init_callback_generate(forest, init)
    replace_callback =
        isnothing(replace) ? C_NULL : _replace_callback_generate(forest, replace)

    _balance_ext(forest, connect, init_callback, replace_callback)

    return
end

@inline function _partition_ext(forest::Pxest{4}, allow_for_coarsening, weight)
    p4est_partition_ext(forest, allow_for_coarsening, weight)
end

@inline function _partition_ext(forest::Pxest{8}, allow_for_coarsening, weight)
    p8est_partition_ext(forest, allow_for_coarsening, weight)
end

@inline function _partition_lnodes(forest::Pxest{4}, ghost, degree, allow_for_coarsening)
    p4est_partition_lnodes(forest, ghost, degree, allow_for_coarsening)
end

@inline function _partition_lnodes(forest::Pxest{8}, ghost, degree, allow_for_coarsening)
    p8est_partition_lnodes(forest, ghost, degree, allow_for_coarsening)
end

function partition!(
    forest::Pxest{X};
    ghost = nothing,
    lnodes_degree = nothing,
    allow_for_coarsening = false,
    weight = nothing,
) where {X}
    if !isnothing(lnodes_degree)
        if isnothing(ghost)
            ghost = ghostlayer(forest)
        end
        _partition_lnodes(forest, ghost, lnodes_degree, allow_for_coarsening)
    else
        if isnothing(weight)
            weight_callback = C_NULL
        else
            weight_callback = _weight_callback_generate(forest, weight)
        end
        _partition_ext(forest, allow_for_coarsening, weight_callback)
    end

    return
end
