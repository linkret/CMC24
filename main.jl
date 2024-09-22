# import LinearAlgebra : norm # doesnt work for some reason, so we have to use the whole module ???
using LinearAlgebra
using Random
using Profile
using ProfileView

include("cmc24.jl")

MIRRORS = 8
PULL_OUT_L = 0.1
PULL_OUT_M = 1.2 # probably too extreme, need to implement this in another way

"""
    generate_segment()

    Returns ((v, e), length) where v is the starting point, e is the direction, and length is the length of the ray.
"""
function generate_segment()
    v = [0, 0]
    while point_in_block(temple, v)
        x = rand() * 19.0 + 1
        y = rand() * 19.0 + 1
        v = [x, y]
    end
    
    # Randomly select an initial angle as e = [cos(α), sin(α)]
    angle = rand() * 360.0 * π / 180
    e = [cos(angle), sin(angle)]
    
    ray = (v, e)
    
    # this method call is a bit slow
    dist = temple_ray_intersection(temple, ray) - PULL_OUT_L # subtract to avoid colliding Lamp with Temple blocks
    collision_point = ray[1] + ray[2] * dist

    rev_ray = (v, -e)
    rev_dist = temple_ray_intersection(temple, rev_ray) - PULL_OUT_M # subtract to avoid colliding Mirror with Temple blocks
    rev_collision_point = rev_ray[1] + rev_ray[2] * rev_dist
    
    return ((collision_point, -e), dist + rev_dist, rev_collision_point)
end

function generate_long_segment(iter_cnt::Int = 200)
    ray, length, endpoint = generate_segment()
    for i in 1:iter_cnt
        new_ray, new_length, new_endpoint = generate_segment()
        if new_length > length
            ray = new_ray
            length = new_length
            endpoint = new_endpoint
        end
    end
    return (ray, length, endpoint)
end

"""
    longest_segment_from_point(v::Array{Float64, 1}, rays, banned_angle::Float64 = -1.0, banned_angle_range::Float64 = 0.20)

    Returns ((v, e), length, endpoint) where v is the starting point, e is the direction, length is the length of the ray, and endpoint is the point of collision.
    The ray is generated from the point v, and the angle of the ray is chosen to maximize the length of the ray.
    The arguemnt rays is used to avoid generating rays that are too close to the previous rays.
    The angle of the ray is chosen from the range [0, 2π) excluding the banned_angle ± banned_angle_range.
    Default banned_angle_range is around 15 degrees (0.20 rad).
"""
function longest_segment_from_point(v::Array{Float64, 1}, rays, banned_angle::Float64 = -1.0, banned_angle_range::Float64 = 0.2)
    if length(v) != 2
        throw(ArgumentError("Input vector v must have length 2"))
    end

    max_length = -1000
    max_ray = ([0, 0], [0, 0])
    max_endpoint = [0, 0]

    # TODO: can make a circular interval radian sweep, where each 1m segment belonging to a block is converted to an interval of angles with a distance
    # Then, we compute intervals of the closest-segments so we know it for each and every angle
    # We can use this to avoid any calls to temple_ray_intersection, which seems to be the bottleneck
    # This could allow us to try even more angles (finer angle-step), or more starting-points

    # radians from 0 to 2π, steps of 1 
    for e in 0:359
        a = deg2rad(e)
        
        if banned_angle != -1.0
            if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                continue
            end
        end
        
        ray = (v, [cos(a), sin(a)])

        intersections = 0
        for ray2 in rays[1:end-1] # end-1 because we surely intersect with the last ray
            r = ray_ray_intersection(ray, ray2) # could try segment_segment_intersection, might be faster, who cares
            if r != (4, 0, 0) && r != (2, 0, 0) # 2 and 4 mean no intersection
                x, y = r[2], r[3]
                if x >= 0 && x <= 20 && y >= 0 && y <= 20 && !point_in_block(temple, [x, y])
                    intersections += 1
                end
            end
        end

        dist = temple_ray_intersection(temple, ray) - PULL_OUT_M # subtract to avoid colliding with Temple blocks
        collision_point = ray[1] + ray[2] * dist

        score_noise = 0 # rand() * 0.5 - 0.25 # seems to not help. And bigger noise is even worse
        score = dist - intersections * 1.2 + score_noise # penalty for every intersection, plus random noise
        if score > max_length
            max_length = score
            max_ray = (v, ray[2])
            max_endpoint = collision_point
        end
    end
    
    return (max_ray, max_length, max_endpoint)
end

function place_mirror(v::Array{Float64, 1}, e::Float64)
    return (v[1] - cos(e) * 0.25, v[2] - sin(e) * 0.25) # 0.25 is the distance from the endpoint to the mirror edges
end

function generate_greedy_solution()
    rays = []

    ray, len, endpoint = generate_long_segment()
    push!(rays, ray)

    #tinfo = @timed begin
    for i in 1:MIRRORS
        # calculate angle from endpoint to the starting point of the previous ray
        banned_angle = atan(ray[1][2] - endpoint[2], ray[1][1] - endpoint[1])
        ray, len, endpoint = longest_segment_from_point(endpoint, rays, banned_angle)
        push!(rays, ray)
    end
    # end # @time
    # println("Time to greedily generate: ", tinfo.time, " s")

    my_solution = Matrix{Float64}(undef, 0, 3)

    lamp_angle = atan(rays[1][2][2], rays[1][2][1])
    my_solution = vcat(my_solution, [rays[1][1][1] rays[1][1][2] lamp_angle])

    # Compute the angles for the mirrors
    for i in 1:(length(rays) - 1)
        end_point = rays[i + 1][1]
        direction1 = rays[i][2]
        direction2 = rays[i + 1][2]

        angle_between_rays = atan(direction2[2], direction2[1]) - atan(direction1[2], direction1[1])
        mirror_angle = atan(direction1[2], direction1[1]) + angle_between_rays / 2
        
        if mirror_angle < 0
            mirror_angle += 2π
        end
        
        mirror = place_mirror(end_point, mirror_angle)
        my_solution = vcat(my_solution, [mirror[1] mirror[2] mirror_angle])
    end

    return my_solution
end

function calculate_solution_length(solution)
    sum = 0
    for i in 1:MIRRORS
        cur_point = solution[i, 1:2]
        nxt_point = solution[i + 1, 1:2]
        dist = norm(nxt_point - cur_point)
        sum += dist
    end
    return sum
end

# Random.seed!(3) # for reproducibility in Debug

sum_scores = 0.0
best_score = 0.0
num_scores = 0.0
num_valid_scores = 0.0

# Profile.clear()
# @profile begin
# elapsed_time = @elapsed begin

while true
    global sum_scores, num_scores, num_valid_scores, best_score
    my_solution = generate_greedy_solution()
    length = calculate_solution_length(my_solution)
    println("Length: ", length) # try to figure out if there is a strong correlation between the length and the score
    
    score = evaluate_solution(my_solution)
    
    num_scores += 1
    if score > 0 # could use > 5 to remove solutions where the Ray immediately hits a wrong mirror and then a wall
        best_score = max(best_score, score)
        sum_scores += score
        num_valid_scores += 1
    end
    
    println("Average score: ", sum_scores / num_valid_scores, " Percent valid scores: ", 100.0 * num_valid_scores / num_scores, " Best score: ", best_score)

    # if num_valid_scores > 5 # break for the sake of profiling
    #     break
    # end
end

rm(fplot1)

# end # elapsed_time
# # ProfileView.view()
# println("Elapsed time: ", elapsed_time)
# end # profile
# Profile.print()