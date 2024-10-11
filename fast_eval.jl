using Images
using ColorTypes

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

const Direction = Point # Type Alias for 2D direction vector

const downscale_factor = 5
const resolution::Int = 3000 / downscale_factor # 600x600 pixels for downscale_factor=5, 1500x1500 for df=2, 3000x3000 for df=1
const pixels::Array{UInt8, 2} = fill(UInt8(1), resolution, resolution)
# 0 = white, 255 = blocked(wall), all other values = red
const total_white_pixels = Ref{Int}(resolution^2)
const current_red_pixels = Ref{Int}(0)

function coord_to_pixel(coord::Float64)::Float64
    # coordinates are floats that go from [0,20], but pixels are ints from [1,resolution]
    return resolution * coord / 20 + 1 # originally was round()
end

function draw_rectangle!(corners::Vector{Point}, color::Int, matrix::Matrix{UInt8})::Nothing
    if length(corners) != 4
        error("corners must have 4 elements")
    end

    # Convert with coord_to_pixel():
    corners = [Point(coord_to_pixel(c[1]), coord_to_pixel(c[2])) for c in corners]

    # Calculate the bounding box of the rectangle
    min_x = floor(Int, minimum(first.(corners)))
    max_x = ceil(Int, maximum(first.(corners)))
    min_y = floor(Int, minimum(last.(corners)))
    max_y = ceil(Int, maximum(last.(corners)))

    block_width = floor(Int, 1.0 / 20 * resolution) # 1.0 meters in pixels

    for x in max(1, min_x):min(resolution, max_x)
        y = max(1, min_y)
        while y <= min(resolution, max_y)
            if matrix[x, y] == 0
                # Snap y to the next multiple of block_width
                y = ceil(Int, y / block_width) * block_width + 1
                continue
            end

            if is_inside_quadrilateral((x, y), corners)
                if color == 0
                    matrix[x, y] = color
                else
                    matrix[x, y] += color # should be careful never to -1 a white pixel or it will underflow
                end
            end

            y += 1
        end
    end
end


function draw_rectangle!(corners::Vector{Point}, color::Int)::Nothing
    global current_red_pixels, total_white_pixels, pixels

    if length(corners) != 4
        error("corners must have 4 elements")
    end

    # Convert with coord_to_pixel():
    corners = [Point(coord_to_pixel(c[1]), coord_to_pixel(c[2])) for c in corners]

    # Calculate the bounding box of the rectangle
    min_x = floor(Int, minimum(first.(corners)))
    max_x = ceil(Int, maximum(first.(corners)))
    min_y = floor(Int, minimum(last.(corners)))
    max_y = ceil(Int, maximum(last.(corners)))

    block_width = floor(Int, 1.0 / 20 * resolution) # 1.0 meters in pixels

    for x in max(1, min_x):min(resolution, max_x)
        y = max(1, min_y)
        while y <= min(resolution, max_y)
            if pixels[x, y] == 0
                # Snap y to the next multiple of block_width
                y = ceil(Int, y / block_width) * block_width + 1
                continue
            end

            if is_inside_quadrilateral((x, y), corners)
                current_red_pixels[] -= (pixels[x, y] > 1)

                if color == 0 # black
                    if pixels[x, y] == 1 # white
                        total_white_pixels[] -= 1 # WARNING: setting a rectangle to White will not increase this
                    end
                    pixels[x, y] = color
                else
                    pixels[x, y] += color # should be careful never to -1 a white pixel or it will underflow
                end

                current_red_pixels[] += (pixels[x, y] > 1)
            end

            y += 1
        end
    end
end

function draw_circle(x1::Float64, y1::Float64, radius::Float64, color::Int, matrix::Matrix{UInt8})::Nothing
    x1, y1 = coord_to_pixel(x1), coord_to_pixel(y1)
    radius = resolution * radius / 20

    min_x = floor(Int, max(1, x1 - radius))
    max_x = ceil(Int, min(resolution, x1 + radius))
    min_y = floor(Int, max(1, y1 - radius))
    max_y = ceil(Int, min(resolution, y1 + radius))

    for x in min_x:max_x
        for y in min_y:max_y
            if matrix[x, y] == 0
                continue
            end

            if (x - x1)^2 + (y - y1)^2 <= radius^2
                matrix[x, y] += color
            end
        end
    end
end


function draw_circle(x1::Float64, y1::Float64, radius::Float64=1.0, color::Int=1)::Nothing
    global current_red_pixels, pixels

    x1, y1 = coord_to_pixel(x1), coord_to_pixel(y1)
    radius = resolution * radius / 20

    min_x = floor(Int, max(1, x1 - radius))
    max_x = ceil(Int, min(resolution, x1 + radius))
    min_y = floor(Int, max(1, y1 - radius))
    max_y = ceil(Int, min(resolution, y1 + radius))

    for x in min_x:max_x
        for y in min_y:max_y
            if pixels[x, y] == 0
                continue
            end

            if (x - x1)^2 + (y - y1)^2 <= radius^2
                current_red_pixels[] -= (pixels[x, y] > 1)

                pixels[x, y] += color

                current_red_pixels[] += (pixels[x, y] > 1)
            end
        end
    end
end

function is_inside_quadrilateral(point::Tuple{Number, Number}, corners::Vector{Point})::Bool
    x, y = point
    winding_number = 0

    for i in 1:4
        p1 = corners[i]
        p2 = corners[mod1(i + 1, 4)]

        if p1.y <= y
            if p2.y > y && (p2.x - p1.x) * (y - p1.y) - (x - p1.x) * (p2.y - p1.y) > 0
                winding_number += 1
            end
        else
            if p2.y <= y && (p2.x - p1.x) * (y - p1.y) - (x - p1.x) * (p2.y - p1.y) < 0
                winding_number -= 1
            end
        end
    end

    return winding_number != 0
end

function draw_rectangle_around_line(x1::Float64, y1::Float64, x2::Float64, y2::Float64, color::Int=1, matrix::Matrix{UInt8} = pixels)::Nothing
    # light_width = 1.0
    angle = atan(y2 - y1, x2 - x1) # angle from p1 to p2
    base_angle = angle + pi / 2 # angle from p1 to the left rectangle corner
    sinb, cosb = sin(base_angle), cos(base_angle)
    p1l = Point(x1 + cosb, y1 + sinb) # actually cosb*light_width, sinb*light_width, but it's 1.0 meters
    p1r = Point(x1 - cosb, y1 - sinb)
    p2l = Point(x2 + cosb, y2 + sinb)
    p2r = Point(x2 - cosb, y2 - sinb)
    if matrix === pixels
        draw_rectangle!([p1l, p1r, p2r, p2l], color)
    else
        draw_rectangle!([p1l, p1r, p2r, p2l], color, matrix)
    end
end

function draw_rectangle_around_line(p1::Point, p2::Point, color::Int=1, matrix::Matrix{UInt8} = pixels)::Nothing
    draw_rectangle_around_line(p1[1], p1[2], p2[1], p2[2], color, matrix)
end

function draw_ray(x1::Float64, y1::Float64, x2::Float64, y2::Float64,
    color::Int=1, second_circle::Bool=true, matrix::Matrix{UInt8} = pixels)::Nothing
    if matrix === pixels
        draw_rectangle_around_line(x1, y1, x2, y2, color)
        draw_circle(x1, y1, 1.0, color)
        if second_circle
            draw_circle(x2, y2, 1.0, color)
        end
    else
        draw_rectangle_around_line(x1, y1, x2, y2, color, matrix)
        draw_circle(x1, y1, 1.0, color, matrix)
        if second_circle
            draw_circle(x2, y2, 1.0, color, matrix)
        end
    end
end

function draw_ray(p1::Point, p2::Point, color::Int=1, second_circle::Bool=true, matrix::Matrix{UInt8} = pixels)::Nothing
    draw_ray(p1[1], p1[2], p2[1], p2[2], color, second_circle, matrix)
end

function draw_block(x1::Float64, y1::Float64)::Nothing
    draw_rectangle!([Point(x1, y1), Point(x1+1, y1), Point(x1+1, y1+1), Point(x1, y1+1)], 0)
end

function draw_temple(temple)::Nothing
    for block in temple.blocks
        draw_block(block.v1[1], block.v1[2])
    end
end

function reset(matrix::Matrix{UInt8} = pixels)::Nothing
    for x in 1:resolution
        for y in 1:resolution
            if matrix[x, y] != 0
                matrix[x, y] = 1
            end
        end
    end
    return
end

function reset()::Nothing
    global pixels, current_red_pixels
    for x in 1:resolution
        for y in 1:resolution
            if pixels[x, y] != 0
                pixels[x, y] = 1
            end
        end
    end
    current_red_pixels[] = 0
    return
end

function fast_score()::Float64
    return 100.0 * current_red_pixels[] / total_white_pixels[] + 0.4 # extra just in case, to still try official evaluate() if we're pretty close
end

function slow_score(matrix1::Matrix{UInt8}, matrix2::Matrix{UInt8})::Float64
    return 100.0 * sum((matrix1 .> 1) .| (matrix2 .> 1)) / total_white_pixels[] + 0.4
end

# These functions work fine, but we don't really need them yet:
function pixel_to_color(value)
    if value == 0
        return RGB(1.0, 1.0, 1.0)  # Black
    elseif value == 1
        return RGB(0.0, 0.0, 0.0)  # White
    else
        return RGB(1.0, 0.0, 0.0)  # Red
    end
end

function draw_pixels_png()
    image = colorview(RGB, [pixel_to_color(pixels[j, i]) for i in 1:size(pixels, 1), j in 1:size(pixels, 2)])
    save("output_image.png", image)
end
