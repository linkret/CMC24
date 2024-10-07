# Computational Modeling Challenge 2024
# CMC24 solution evaluation script
# Author: Hrvoje Abraham, hrvoje.abraham@avl.com

using FileIO
using Plots; gr()
using Measures
using DelimitedFiles

import Base: hash

const temple_string =
#1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9  0
"O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O
 O  .  .  .  .  O  .  .  .  .  .  .  .  .  O  .  .  .  .  O
 O  .  .  .  .  .  .  .  O  .  .  O  .  .  .  .  .  .  .  O
 O  .  .  .  .  .  O  .  .  .  .  .  .  O  .  .  .  .  .  O
 O  .  .  O  .  .  .  .  .  O  O  .  .  .  .  .  O  .  .  O
 O  O  .  .  .  .  .  .  .  O  O  .  .  .  .  .  .  .  O  O
 O  .  .  .  O  .  .  .  .  .  .  .  .  .  .  O  .  .  .  O
 O  .  .  .  .  .  .  O  .  .  .  .  O  .  .  .  .  .  .  O
 O  .  .  .  .  .  .  .  .  O  O  .  .  .  .  .  .  .  .  O
 O  .  O  .  .  O  O  .  .  .  .  .  .  O  O  .  .  O  .  O
 O  .  O  .  .  O  O  .  .  .  .  .  .  O  O  .  .  O  .  O
 O  .  .  .  .  .  .  .  .  O  O  .  .  .  .  .  .  .  .  O
 O  .  .  .  .  .  .  O  .  .  .  .  O  .  .  .  .  .  .  O
 O  .  .  .  O  .  .  .  .  .  .  .  .  .  .  O  .  .  .  O
 O  O  .  .  .  .  .  .  .  O  O  .  .  .  .  .  .  .  O  O
 O  .  .  O  .  .  .  .  .  O  O  .  .  .  .  .  O  .  .  O
 O  .  .  .  .  .  O  .  .  .  .  .  .  O  .  .  .  .  .  O
 O  .  .  .  .  .  .  .  O  .  .  O  .  .  .  .  .  .  .  O
 O  .  .  .  .  O  .  .  .  .  .  .  .  .  O  .  .  .  .  O
 O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O  O"

const block_size = 1
const mirror_length = 0.5
const light_halfwidth = 1
const ε = 1e-12

# TODO: seperate into another "header" .jl script
struct Point 
    x::Float64
    y::Float64
end

function Point()
    return Point(0., 0.)
end

function Point(arr::Vector)
    return Point(arr[1], arr[2])
end

import Base.first

function first(p::Point)
    return p[1]
end

import Base.last

function last(p::Point)
    return p[2]
end

import Base.length

function length(p::Point)
    return 2
end

import Base.getindex

function getindex(p::Point, i::Int64)
    if i == 1
        return p.x
    elseif i == 2
        return p.y
    else
        return 0. # TODO: throw Error
    end
end

import Base.+

function +(p1::Point, p2::Point)
    return Point(p1.x + p2.x, p1.y + p2.y)
end

import Base.-

function -(p1::Point, p2::Point)
    return Point(p1.x - p2.x, p1.y - p2.y) # TODO: verify its not the other way around
end

import Base.*

function *(p::Point, k::Number)
    return Point(p.x * k, p.y * k)
end

function *(k::Number, p::Point)
    return *(p, k)
end

function hash(p::Point, h::UInt)::UInt
    h = hash(p.x, h)
    h = hash(p.y, h)
    return h
end

const Direction = Point # Type Alias for 2D direction vector

struct Ray
    point::Point
    direction::Direction
end

struct Segment
    point::Point
    length::Float64
    angle::Float64
end

struct Block
    v1::Point # bottom left corner
    v2::Point
    v3::Point # up right corner
    v4::Point

    s1::Segment
    s2::Segment
    s3::Segment
    s4::Segment
end

function hash(b::Block, h::UInt)::UInt
    h = hash(b.v1, h)
    h = hash(b.v2, h)
    h = hash(b.v3, h)
    h = hash(b.v4, h)
    # Should be enough to hash the block by its vertices
    return h
end

struct Temple
    blocks::Set{Block}
    shape::Tuple{Int64, Int64}
    size::Tuple{Int64, Int64}
    grid::Vector{Vector{Union{Block, Nothing}}}
end

function hash(t::Temple, h::UInt)::UInt
    h = hash(t.blocks, h)
    h = hash(t.shape, h)
    return h
end

struct Mirror 
    v1::Point
    v2::Point
    s::Segment
    α::Float64
    e::Direction
    n::Direction
end

struct Lamp
    v::Point
    α::Float64
    e::Direction
end

struct Path
    points::Vector{Point}
    directions::Vector{Direction}
end

"""Float64 infinity"""
const ∞ = Inf

"""
    ⋅(v, w)

Dot product of two 2D vectors.
"""
function ⋅(v, w)::Float64
    return v[1] * w[1] + v[2] * w[2]
end

"""
    ×(v, w)

Cross product of two 2D vectors.
"""
function ×(v, w)::Float64
    return v[1] * w[2] - v[2] * w[1]
end

"""Last 12 digits of hexadecimal format of the input integer."""
function hex12(x::Integer)::String
    return last(string(x, base=16), 12)
end

"""
    Convert integer into string with digit grouping.

    E.g. 1234567 => "1,234,567"
"""
function commas(num::Integer)::String
    str = string(num)
    return replace(str, r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
end

function load_temple(temple_string::String, block_size::Int64)::Temple
    # println(stderr, " " * temple_string)
    rows = split(replace(strip(temple_string), " " => ""), '\n')
    temple_shape = length(rows[1]), length(rows)
    grid = []

    temple = Set{Block}()
    for (j, row) ∈ enumerate(rows)
        grid_row = Vector{Union{Block, Nothing}}()
        for (i, c) ∈ enumerate(row)
            if c == 'O'
                x = (i - 1) * block_size
                y = temple_shape[2] - j * block_size
                
                v1 = Point(x, y)
                v2 = Point(x + block_size, y)
                v3 = Point(x + block_size, y + block_size)
                v4 = Point(x, y + block_size)

                block = Block(
                    v1,
                    v2,
                    v3,
                    v4,
                    Segment(v1, block_size, 0),
                    Segment(v2, block_size, π/2),
                    Segment(v3, block_size, π),
                    Segment(v4, block_size, 3π/2),
                )
                
                push!(temple, block)
                push!(grid_row, block)
            else 
                push!(grid_row, nothing)
            end
        end
        pushfirst!(grid, grid_row)
    end

    println(stderr, "The temple of size $temple_shape is loaded.")

    return Temple(
        temple,
        temple_shape,
        block_size .* temple_shape,
        grid
    )
end

function load_solution(cmc24_solution::Matrix{Float64}, mirror_length::Float64)::Tuple{Lamp, Vector{Mirror}}
    if size(cmc24_solution) ≠ (9, 3)
        println(stderr, "ERROR! The solution isn't 9x3 size matrix.")
        finalize()
        return ()
    end

    if !(eltype(cmc24_solution) <: Number)
        println(stderr, "ERROR! The solution contains non-numerical inputs.")
        finalize()
        return ()
    end

    try
        cmc24_solution = float(cmc24_solution)
    catch
        println(stderr, "ERROR! The solution can't be converted to double precision floating point format.")
        finalize()
        return ()
    end

    # preprocess the lamp
    α = cmc24_solution[1, 3]
    lamp = Lamp(
        Point(cmc24_solution[1, 1:2]),
        α,
        Direction(cos(α), sin(α))
    )

    # preprocess the mirrors
    mirrors = Vector{Mirror}()
    for m ∈ 1 : 8
        α = cmc24_solution[m + 1, 3]
        
        v = Point(cmc24_solution[m + 1, 1:2])
        e = Direction(cos(α),  sin(α))
        n = Direction(-sin(α), cos(α))  # normal

        mirror = Mirror(
            v,
            v + mirror_length * e,
            Segment(v, mirror_length, α),
            α,
            e,
            n,
        )

        push!(mirrors, mirror)
    end

    # println(stderr, "The solution is loaded.")

    return (lamp, mirrors)
end

function point_in_block(point::Point, block::Union{Block, Nothing})::Bool
    return block ≠ nothing && block.v1.x .≤ point.x .≤ block.v3.x && block.v1.y .≤ point.y .≤ block.v3.y;
end

function point_in_temple(temple::Temple, point::Point)::Bool
    #for cell ∈ temple.blocks
        cell = block_from_point(temple, point)
        # if the point is within bottom-left and top-right vertex
        if point_in_block(point, cell)
            return true
        end
    #end

    return false
end

function block_from_point(temple::Temple, point::Point)::Union{Block, Nothing}
    x = floor(Int, point[1])
    y = floor(Int, point[2])
    if x < 0 || y < 0 || x ≥ temple.shape[1] || y ≥ temple.shape[2]
        return nothing
    end
    return temple.grid[y + 1][x + 1]
end

function point_sector(temple::Temple, point::Point)::Int
    sx = floor(Int, 3 * point[1] / temple.size[1])
    sy = floor(Int, 3 * point[2] / temple.size[2])

    return 3 * sy + sx + 1
end

function ray_ray_intersection(ray1::Ray, ray2::Ray)::Tuple{Int, Float64, Float64}
    p = ray1.point
    r = ray1.direction
    
    q = ray2.point
    s = ray2.direction

    rs = r × s
    qpr = (q - p) × r

    # CASE 1 - rays are collinear and maybe overlap
    if (rs == 0) && (qpr == 0)
        t0 = (q - p) ⋅ r / (r ⋅ r)
        t1 = (q + s - p) ⋅ r / (r ⋅ r)
        return (1, t0, t1)
    end

    # CASE 2 - rays are parallel so they don't intersect
    if (rs == 0) && (qpr ≠ 0)
        return (2, 0.0, 0.0)
    end

    # CASE 3 - rays intersect
    qps = (q - p) × s
    t = qps / rs
    u = qpr / rs
    if (rs ≠ 0) && (t ≥ 0) && (u ≥ 0)
        return (3, t, u)
    end

    # CASE 4 - rays don't intersect
    return (4, 0.0, 0.0)
end

function ray_segment_intersection(ray::Ray, segment::Segment)::Tuple{Int, Float64, Float64}
    p = ray.point
    r = ray.direction
    
    q = segment.point
    l = segment.length
    β = segment.angle

    s = l * Point(cos(β), sin(β))

    (case, t, u) = ray_ray_intersection(ray, Ray(q, s))

    # CASE 1 - No intersection
    if case == 1 && t < 0 && u < 0            return (1, 0., 0.) end
    if case == 2                              return (1, 0., 0.) end
    if case == 3 && (t ≤ 0 || u < 0 || u > 1 + ε) return (1, 0., 0.) end
    if case == 4                              return (1, 0., 0.) end

    # CASE 2 - Ray and segment are collinear and they intersect
    if case == 1
        if t > 0 && u ≥ 0 return (2, min(t, u), 0.) end
        if t ≥ 0          return (2, t, 0.)         end
        if u ≥ 0          return (2, 0., 0.)        end
    end

    # CASE 3 - Ray and segment intersect in ordinary way
    return (3, t, u)
end

function segment_segment_intersection(segment1::Segment, segment2::Segment)::Bool
    p = segment1.point
    la = segment1.length
    α = segment1.angle
    
    q = segment2.point
    lb = segment2.length
    β = segment2.angle

    r = la * Point(cos(α), sin(α))
    s = lb * Point(cos(β), sin(β))

    (case, t, u) = ray_ray_intersection(Ray(p, r), Ray(q, s))

    if case == 1 && r ⋅ s > 0 && t ≤ u && (0 ≤ t ≤ 1 || 0 ≤ u ≤ 1)
        return true
    end

    if case == 1 && r ⋅ s < 0 && t ≥ u && (0 ≤ t ≤ 1 || 0 ≤ u ≤ 1)
        return true
    end

    if case == 2
        return false
    end

    if case == 3 && (0 ≤ t ≤ 1 && 0 ≤ u ≤ 1)
        return true
    end

    if case == 4
        return false
    end

    return false
end

function segment_block_intersection(segment::Segment, block::Block)::Bool
    return any((
        #point_in_block(segment[1], block),
        #point_in_block(segment[1] + segment[2] * [cos(segment[3]), sin(segment[3])], block),
        segment_segment_intersection(segment, block.s1),
        segment_segment_intersection(segment, block.s2),
        segment_segment_intersection(segment, block.s3),
        segment_segment_intersection(segment, block.s4),
    ))
end

function temple_segment_intersection(temple::Temple, segment::Segment)::Bool
    return any(segment_block_intersection(segment, block) for block ∈ temple.blocks)
end

function euclidean_distance(p1::Point, p2::Point)::Float64
    return hypot(p1[1] - p2[1], p1[2] - p2[2])
end

function temple_ray_intersection(temple::Temple, ray::Ray)::Float64
    t_min = ∞
    for block ∈ temple.blocks
        # t_approx = min(
        #     euclidean_distance(ray.point, block.v1),
        #     euclidean_distance(ray.point, block.v2),
        #     euclidean_distance(ray.point, block.v3),
        #     euclidean_distance(ray.point, block.v4)
        # )

        # if t_approx > t_min
        #     continue # not 100% precise when we are for example directly below a block, but okay. The early-exit provides around a 30% speedup
        # end

        # only check the blocks that are in the direction of the ray's origin. Over 50% speedup over checking all 4 block segments

        if ray.point[2] <= block.s1.point[2]
            (case, t, u) = ray_segment_intersection(ray, block.s1)
            if (case == 2 || case == 3) && (t < t_min) && (t > ε)
                t_min = t
            end
        end

        if ray.point[1] >= block.s2.point[1]
            (case, t, u) = ray_segment_intersection(ray, block.s2)
            if (case == 2 || case == 3) && (t < t_min) && (t > ε)
                t_min = t
            end
        end

        if ray.point[2] >= block.s3.point[2]
            (case, t, u) = ray_segment_intersection(ray, block.s3)
            if (case == 2 || case == 3) && (t < t_min) && (t > ε)
                t_min = t
            end
        end

        if ray.point[1] <= block.s4.point[1]
            (case, t, u) = ray_segment_intersection(ray, block.s4)
            if (case == 2 || case == 3) && (t < t_min) && (t > ε)
                t_min = t
            end
        end
    end

    if t_min == Inf
        println("ERROR, INFINITE DISTANCE")
    end

    return t_min
end

function check_solution(temple::Temple, lamp::Lamp, mirrors::Vector{Mirror})::Bool
    # check the lamp is within the temple
    if !(all(0 .≤ lamp.v.x .≤ temple.size) && all(0 .≤ lamp.v.y .≤ temple.size))
        println(stderr, "ERROR! The lamp isn't placed within temple limits which is of size $(temple.size).")
        finalize(temple, lamp, mirrors)
        return false
    end

    # check mirrors' ends are within the temple
    if !all((all(0 .≤ mirror.v1.x .≤ temple.size) && all(0 .≤ mirror.v1.y .≤ temple.size)) for mirror ∈ mirrors)
        println(stderr, "ERROR! Some mirror isn't placed within temple of size $(temple.size).")
        finalize(temple, lamp, mirrors)
        return false
    end

    if !all((all(0 .≤ mirror.v2.x .≤ temple.size) && all(0 .≤ mirror.v2.y .≤ temple.size)) for mirror ∈ mirrors)
        println(stderr, "ERROR! Some mirror isn't placed within temple of size $(temple.size).")
        finalize(temple, lamp, mirrors)
        return false
    end
    
    # check the lamp isn't in some building block
    if point_in_temple(temple, lamp.v)
        println(stderr, "ERROR! Lamp is placed in a building block.")
        finalize(temple, lamp, mirrors)
        return false
    end
    
    # check some mirror end isn't in some building block
    for (m, mirror) ∈ enumerate(mirrors)
        if point_in_temple(temple, mirror.v1) || point_in_temple(temple, mirror.v2)
            println(stderr, "ERROR! Mirror $m has one of its ends inside a building block.")
            finalize(temple, lamp, mirrors)
            return false
        end
    end

    # check some mirror doesn't overlap with some building block
    for (m, mirror) ∈ enumerate(mirrors)
        if temple_segment_intersection(temple, mirror.s)
            println(stderr, "ERROR! Mirror $m intersects with a building block.")
            finalize(temple, lamp, mirrors)
            return false
        end
    end
    
    # check if some mirrors intersect
    for (m1, mirror1) ∈ enumerate(mirrors[1:end-1]), (m2, mirror2) ∈ enumerate(mirrors[m1+1:end])
        if segment_segment_intersection(mirror1.s, mirror2.s)
            println(stderr, "ERROR! Mirrors $m1 & $(m1+m2) intersect.")
            finalize(temple, lamp, mirrors)
            return false
        end
    end
    
    # println(stderr, "The solution geometry is correct.")

    return true
end

function raytrace(temple::Temple, lamp::Lamp, mirrors::Vector{Mirror})::Path
    local hit_mirror

    path = Path(
        [lamp.v],
        []
    )

    ray = Ray(
        lamp.v,
        lamp.e
    )

    hit_mirrors = Vector{Int}()
    while true
        # check if ray can hit some mirror
        t_mirror = ∞
        for (m, mirror) ∈ enumerate(mirrors)
            (case, t, u) = ray_segment_intersection(ray, mirror.s)
            if ((case == 2) || (case == 3)) && (t < t_mirror) && (t > ε)
                t_mirror = t
                hit_mirror = mirror
                push!(hit_mirrors, m)
            end
        end

        # check where ray would hit the temple
        t_temple = temple_ray_intersection(temple, ray)

        # closest hit point
        t = min(t_mirror, t_temple)
        hitting_point = ray.point + t * ray.direction
        push!(path.directions, ray.direction)
        push!(path.points, hitting_point)

        # ray hit a mirror, calculate new direction
        if t_mirror < t_temple
            ray = Ray(
                hitting_point,
                ray.direction - 2 * (ray.direction ⋅ hit_mirror.n) * hit_mirror.n
            )
            continue
        end

        # ray hit the temple
        break
    end

    return path
end

function consistent_hash(args...) # probably dont need this
    # Replace `nothing` with a consistent placeholder value
    args = map(x -> x === nothing ? :nothing_placeholder : x, args)
    return hash(args)
end

function cmc24_plot(
    temple::Temple; 
    lamp::Union{Lamp, Nothing}=nothing, 
    mirrors::Union{Vector{Mirror}, Nothing}=nothing, 
    path::Union{Path, Nothing}=nothing, 
    downscale_factor::Float64=1.0
)::String
    plot_scale = 150 / downscale_factor # to speed up solution evaluation
    plot_size = plot_scale .* temple.shape 
    
    solution_hash = hex12(consistent_hash([temple, lamp, mirrors, path]))
    filename = "cmc24_solution_" * solution_hash * ".png"

    if isfile(filename)
        return filename # It's already plotted
    end

    plot(
        size = plot_size,
        xlims = (0, temple.size[1]),
        ylims = (0, temple.size[2]),
        background_color = RGBA(0.9, 0.87, 0.7, 1),
        label = false,
        showaxis = false,
        grid = false,
        legend = false,
        aspect_ratio = 1,
        bottom_margin = -20mm,
        right_margin = -10mm,
        top_margin = -10mm,
        left_margin = -20mm,
    )
    
    function circleShape(x, y, r, n)
        θ = LinRange(0, 2π, n+1)
        return Shape(x .+ r*cos.(θ), y .+ r*sin.(θ))
    end
    
    # plot the lightened area
    if path ≠ nothing
        # circle parts of the lightened area
        for p ∈ path.points
            plot!(
                circleShape(p[1], p[2], light_halfwidth, 1000),
                color = RGBA(1, 0.7, 0.6, 1),
                linecolor = RGBA(0, 0, 0, 0),
                linewidth = 0,
            )    
        end
        
        # rectangle parts of the lightened area
        for (p1, p2, e) ∈ zip(path.points, path.points[2:end], path.directions)
            n = Direction(e[2], -e[1])
            
            xs = [p1.x - n.x, p2.x - n.x, p2.x + n.x, p1.x + n.x]
            ys = [p1.y - n.y, p2.y - n.y, p2.y + n.y, p1.y + n.y]
            
            plot!(
                Shape(xs, ys),
                color = RGBA(1, 0.7, 0.6, 1),
                linecolor = RGBA(0, 0, 0, 0),
                linewidth = 0,
            )    
        end
    end
    
    # plot the mirrors
    if mirrors ≠ nothing
        for mirror ∈ mirrors
            p1 = mirror.v1
            p2 = mirror.v2
            n = 0.05 * mirror.n

            xs = [p1.x - n.x, p2.x - n.x, p2.x + n.x, p1.x + n.x]
            ys = [p1.y - n.y, p2.y - n.y, p2.y + n.y, p1.y + n.y]

            plot!(
                Shape(xs, ys),
                color = RGBA(0, 0, 1, 1),
                linecolor = RGBA(0, 0, 0, 0),
                linewidth = 0,
            )    
        end
    end

    # plot the ray
    if path ≠ nothing
        plot!(
            first.(path.points),
            last.(path.points),
            linecolor = RGBA(1, 0, 0, 1),
            linewidth = 0.04 * plot_scale,
        )
    end

    # plot the lamp
    if lamp ≠ nothing
        plot!(
            circleShape(lamp.v[1], lamp.v[2], 0.2, 6),
            color = RGBA(0.9, 0, 1, 1),
            linecolor = RGBA(1, 1, 1, 1),
            linewidth = 5,
        )    
    end
    
    # plot the building blocks
    for block ∈ temple.blocks
        p = block.v1
        plot!(
            Shape(
                p.x .+ [0, block_size, block_size, 0],
                p.y .+ [0, 0, block_size, block_size]),
            color = RGBA(0.50, 0.48, 0.47, 1),
            linecolor = RGBA(0, 0, 0, 0),
            linewidth = 0,
        )
    end
    
    savefig(filename)
    
    return filename
end

function evaluate(temple::Temple, path::Path)::Tuple{Int64, Int64, Int64}
    global fplot1, img1
    #fplot1 = cmc24_plot(temple)
    #tinfo = @timed begin
    fplot2 = cmc24_plot(temple, path=path)
    #end
    #println("Time to plot the path: ", tinfo.time, " s")
    
    # img1 = FileIO.load(fplot1)
    img2 = FileIO.load(fplot2) # slow

    # count the total number of the plot pixels
    total = length(img1)

    # count the number of vacant pixels recognized by being bright
    vacant = sum(p.r > 0.7 for p ∈ img1) # can precompute, but its not that slow

    # count the number of pixels changed due to the light ray
    score = sum(p1 ≠ p2 for (p1, p2) ∈ zip(img1, img2))

    # delete image files
    # rm(fplot1)
    rm(fplot2)
    
    return total, vacant, score
end

function finalize(
    temple::Union{Temple, Nothing}=nothing,
    lamp::Union{Lamp, Nothing}=nothing,
    mirrors::Union{Vector{Mirror}, Nothing}=nothing,
    path::Union{Path, Nothing}=nothing
)::Nothing
    if temple ≠ nothing
        # Can uncomment if we want to inspect the image with the invalid solution
        # cmc24_plot(temple, lamp=lamp, mirrors=mirrors, path=path)
    end
    
    # println(0)  # stdout print of a fallback result in a case of early exit

    # exit() # uncomment if we want to stop the script execution
    return
end

function load_solution_file(filename::String)::Tuple{Float64, Matrix{Float64}}
    if !isfile(filename)
        println(stderr, "ERROR! The file $filename doesn't exist.")
        return 0, []
    end

    lines = readlines(filename)
    
    # Extract the float
    score = parse(Float64, lines[1])
    
    # Extract the matrix
    matrix_data = readdlm(IOBuffer(join(lines[2:end], "\n")), Float64) # a bit slow but whatever
    
    return score, matrix_data
end

function evaluate_solution(cmc24_solution::Matrix{Float64})::Float64
    global best_score
    # load the solution
    lamp, mirrors = load_solution(cmc24_solution, mirror_length)
    if !check_solution(temple, lamp, mirrors)
        return 0
    end
    
    # compute the ray path
    path = raytrace(temple, lamp, mirrors)
    
    # evaluate the solution
    total, vacant, score = evaluate(temple, path)
    # println(stderr, "Base plot has $(commas(vacant)) vacant of total $(commas(total)) pixels.")
    score_percent = 100. * score / vacant
    println(stderr, "Your CMC24 score is $(commas(score)) / $(commas(vacant)) = $(100. * score / vacant) %.")
    
    # best_score[1], best_matrix = load_solution_file("best.txt")
    if score_percent > best_score[1]
        println(stderr, "Congratulations! You have a new best score.")
        open("best.txt", "w") do io
            println(io, score_percent)
            writedlm(io, cmc24_solution)
        end
        
        best_score[1] = score_percent
        cmc24_plot(temple, lamp=lamp, mirrors=mirrors, path=path)
    end

    return score_percent
end

const best_solution = load_solution_file("best.txt")
const best_score = [best_solution[1]] # array to be mutable
const temple = load_temple(temple_string, block_size)

# TODO: this precompute can be moved to the point of first-use - there are rare cases we don't need it at all
const fplot1 = cmc24_plot(temple) # precompute the static base plot
const img1 = FileIO.load(fplot1) # preload the static base image

# println("Current best solution: ")
# evaluate_solution(test_solution)

# rm(fplot1) # delete the static base plot # we usually perform this cleanup in main.jl