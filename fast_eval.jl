using LinearAlgebra

resolution = 600 # 600x600 pixels for downscale_factor=5, 1500x1500 for df=2, 3000x3000 for df=1
pixels = fill(UInt8(0), resolution, resolution)
# 0 = white, 255 = blocked(wall), all other values = red
total_white_pixels = 0
current_red_pixels = 0

function coord_to_pixel(coord)
    # coordinates are floats that go from [0,20], but pixels are ints from [1,resolution]
    return round(Int, resolution * coord / 20) + 1 # TODO: try floor()
end

# Function to draw a rectangle at an angle
function draw_rectangle!(corners, color, set=false)
    if length(corners) != 4
        error("corners must have 4 elements")
    end

    # Draw the rectangle by filling in the pixels
    for x in 1:resolution
        for y in 1:resolution
            if pixels[x, y] == 255
                continue
            end

            if is_inside_polygon((x, y), corners)
                if set
                    pixels[x, y] = color
                else
                    pixels[x, y] += color # should be careful never to -1 a white pixel or it will underflow
                end
            end
        end
    end
end

function draw_rectangle_around_line(x1, y1, x2, y2, color)
    # light_width = 1.0
    angle = atan(y2 - y1, x2 - x1) # angle from p1 to p2
    base_angle = angle + pi / 2 # angle from p1 to the left rectangle corner
    sinb, cosb = sin(base_angle), cos(base_angle)
    x1l, y1l = x1 + cosb, y1 + sinb # actually cosb*light_width, sinb*light_width, but it's 1.0 meters
    x1r, y1r = x1 - cosb, y1 - sinb
    x2l, y2l = x2 + cosb, y2 + sinb
    x2r, y2r = x2 - cosb, y2 - sinb
    p1 = [coord_to_pixel(x1l), coord_to_pixel(y1l)]
    p2 = [coord_to_pixel(x1r), coord_to_pixel(y1r)]
    p3 = [coord_to_pixel(x2r), coord_to_pixel(y2r)]
    p4 = [coord_to_pixel(x2l), coord_to_pixel(y2l)]
    draw_rectangle!([p1, p2, p3, p4], color)
end