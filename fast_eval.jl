#using Images
#using ColorTypes
using LinearAlgebra: norm, dot

const Point = Vector{Float64} # from main.jl

const downscale_factor = 5
const resolution::Int = 3000 / downscale_factor # 600x600 pixels for downscale_factor=5, 1500x1500 for df=2, 3000x3000 for df=1
const pixels::Array{UInt8, 2} = fill(UInt8(0), resolution, resolution)
# 0 = white, 255 = blocked(wall), all other values = red
const total_white_pixels = Ref{Int}(resolution^2)
const current_red_pixels = Ref{Int}(0)

function coord_to_pixel(coord::Float64)::Float64
    # coordinates are floats that go from [0,20], but pixels are ints from [1,resolution]
    return resolution * coord / 20 + 1 # originally was round()
end

function draw_rectangle!(corners::Vector{Vector{Float64}}, color::Int)::Nothing
    global current_red_pixels, total_white_pixels, pixels

    if length(corners) != 4
        error("corners must have 4 elements")
    end

    # Convert with coord_to_pixel():
    corners = [[coord_to_pixel(c[1]), coord_to_pixel(c[2])] for c in corners]

    # Calculate the bounding box of the rectangle
    min_x = floor(Int, minimum(first.(corners)))
    max_x = ceil(Int, maximum(first.(corners)))
    min_y = floor(Int, minimum(last.(corners)))
    max_y = ceil(Int, maximum(last.(corners)))

    block_width = floor(Int, 1.0 / 20 * resolution) # 1.0 meters in pixels

    for x in max(1, min_x):min(resolution, max_x)
        y = max(1, min_y)
        while y <= min(resolution, max_y)
            if pixels[x, y] == 255
                # Snap y to the next multiple of block_width
                y = ceil(Int, y / block_width) * block_width + 1
                continue
            end

            # TODO: this looks maybe kinda inefficient tbh
            if is_inside_quadrilateral((x, y), corners) # TODO: don't need to use this check for Blocks, but okay
                current_red_pixels[] -= (pixels[x, y] != 0 && pixels[x, y] != 255)

                if color == 255
                    if pixels[x, y] == 0
                        total_white_pixels[] -= 1 # WARNING: setting a rectangle to White will not increase this
                    end
                    pixels[x, y] = color
                else
                    pixels[x, y] += color # should be careful never to -1 a white pixel or it will underflow
                end

                current_red_pixels[] += (pixels[x, y] != 0 && pixels[x, y] != 255)
            end

            y += 1
        end
    end
end

# function point_to_segment_distance(px::Point, py::Point, x1::Float64, y1::Float64, x2::Float64, y2::Float64)::Float64
#     line_vec = [x2 - x1, y2 - y1]
#     point_vec = [px - x1, py - y1]
#     line_len = norm(line_vec)
#     line_unitvec = line_vec / line_len
#     point_vec_scaled = point_vec / line_len
#     t = dot(line_unitvec, point_vec_scaled)
#     t_clamped = clamp(t, 0.0, 1.0)
#     nearest = [x1, y1] + t_clamped * line_vec
#     return norm([px, py] - nearest)
# end

# function draw_area_around_ray!(x1::Float64, y1::Float64, x2::Float64, y2::Float64, color::Int=1)::Nothing
#     global current_red_pixels, pixels

#     x1, y1 = coord_to_pixel(x1), coord_to_pixel(y1) # TODO: verify if this is correct
#     x2, y2 = coord_to_pixel(x2), coord_to_pixel(y2)
#     radius = resolution * 1.0 / 20

#     min_x = floor(Int, max(1, min(x1 - radius, x2 - radius)))
#     max_x = ceil(Int, min(resolution, max(x1 + radius, x2 + radius)))
#     min_y = floor(Int, max(1, min(y1 - radius, y2 - radius)))
#     max_y = ceil(Int, min(resolution, max(y1 + radius, y2 + radius)))

#     for x in min_x:max_x
#         for y in min_y:max_y
#             if pixels[x, y] == 255
#                 continue
#             end

#             if point_to_segment_distance(x, y, x1, y1, x2, y2) <= radius
#                 current_red_pixels[] -= (pixels[x, y] != 0 && pixels[x, y] != 255)

#                 pixels[x, y] += color

#                 current_red_pixels[] += (pixels[x, y] != 0 && pixels[x, y] != 255)
#             end
#         end
#     end
# end

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
            if pixels[x, y] == 255
                continue
            end

            if (x - x1)^2 + (y - y1)^2 <= radius^2
                current_red_pixels[] -= (pixels[x, y] != 0 && pixels[x, y] != 255)

                pixels[x, y] += color

                current_red_pixels[] += (pixels[x, y] != 0 && pixels[x, y] != 255)
            end
        end
    end
end

function is_inside_quadrilateral(point::Tuple{Number, Number}, corners::Vector{Point})::Bool
    x, y = point
    winding_number = 0

    for i in 1:4
        x1, y1 = corners[i]
        x2, y2 = corners[mod1(i + 1, 4)]

        if y1 <= y
            if y2 > y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0
                winding_number += 1
            end
        else
            if y2 <= y && (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0
                winding_number -= 1
            end
        end
    end

    return winding_number != 0
end

function draw_rectangle_around_line(x1::Float64, y1::Float64, x2::Float64, y2::Float64, color::Int=1)::Nothing
    # light_width = 1.0
    angle = atan(y2 - y1, x2 - x1) # angle from p1 to p2
    base_angle = angle + pi / 2 # angle from p1 to the left rectangle corner
    sinb, cosb = sin(base_angle), cos(base_angle)
    p1l = [x1 + cosb, y1 + sinb] # actually cosb*light_width, sinb*light_width, but it's 1.0 meters
    p1r = [x1 - cosb, y1 - sinb]
    p2l = [x2 + cosb, y2 + sinb]
    p2r = [x2 - cosb, y2 - sinb]
    draw_rectangle!([p1l, p1r, p2r, p2l], color)
end

function draw_rectangle_around_line(p1::Point, p2::Point, color::Int=1)::Nothing
    draw_rectangle_around_line(p1[1], p1[2], p2[1], p2[2], color)
end

function draw_ray(x1::Float64, y1::Float64, x2::Float64, y2::Float64, color::Int=1, second_circle::Bool=true)::Nothing
    draw_rectangle_around_line(x1, y1, x2, y2, color)
    draw_circle(x1, y1, 1.0, color)
    if second_circle
        draw_circle(x2, y2, 1.0, color)
    end
    #draw_area_around_ray!(x1, y1, x2, y2, color) # this shit is way slower, Sadge
end

function draw_ray(p1::Point, p2::Point, color::Int=1, second_circle::Bool=true)::Nothing
    draw_ray(p1[1], p1[2], p2[1], p2[2], color, second_circle)
end

function draw_block(x1::Float64, y1::Float64)::Nothing
    draw_rectangle!([[x1, y1], [x1+1, y1], [x1+1, y1+1], [x1, y1+1]], 255)
end

function draw_temple(temple)::Nothing
    for block in temple.blocks
        draw_block(block.v1[1], block.v1[2])
    end
end

function reset()::Nothing
    global pixels, current_red_pixels
    #pixels = fill(UInt8(0), resolution, resolution)
    for x in 1:resolution
        for y in 1:resolution
            if pixels[x, y] != 255
                pixels[x, y] = 0
            end
        end
    end
    current_red_pixels[] = 0
    return
    # total_white_pixels[] = resolution^2
    # draw_temple(temple)
end

function fast_score()::Float64
    return 100.0 * current_red_pixels[] / total_white_pixels[] + 0.5 # 0.5 just in case, to still try official evaluate() if we're pretty close
end

# These functions work fine, but we don't really need them yet:
# function pixel_to_color(value)
#     if value == 0
#         return RGB(1.0, 1.0, 1.0)  # White
#     elseif value == 255
#         return RGB(0.0, 0.0, 0.0)  # Black
#     else
#         return RGB(1.0, 0.0, 0.0)  # Red
#     end
# end

# function draw_pixels_png()
#     image = colorview(RGB, [pixel_to_color(pixels[j, i]) for i in 1:size(pixels, 1), j in 1:size(pixels, 2)])
#     save("output_image.png", image)
# end