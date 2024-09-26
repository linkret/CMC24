# import LinearAlgebra : norm # doesnt work for some reason, so we have to use the whole module ???
using LinearAlgebra
using Random
using Profile
using ProfileView
using Base.Iterators: drop

include("cmc24.jl")
include("fast_eval.jl")

MIRRORS = 8
PULL_OUT_L = 0.5
PULL_OUT_M = 0.5 # 1.2 was probably too extreme

"""
    generate_segment()

    Returns ((v, e), length) where v is the starting point, e is the direction, and length is the length of the ray.
"""
function generate_segment()
    v = [0, 0]
    while point_in_temple(temple, v)
        x = rand() * 19.0 + 1
        y = rand() * 19.0 + 1
        v = [x, y]
    end
    
    # Randomly select an initial angle as e = [cos(α), sin(α)]
    angle = rand() * 2 * π
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

function generate_long_segment(iter_cnt::Int = 100)
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
function longest_segment_from_point(v::Array{Float64, 1}, rays, mirrors, banned_angle::Float64 = -1.0, banned_angle_range::Float64 = 0.2)
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
    for e in 0:1:359
        a = deg2rad(e)
        
        if banned_angle != -1.0
            if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                continue
            end
        end
        
        ray = (v, [cos(a), sin(a)])
        dist = temple_ray_intersection(temple, ray) - PULL_OUT_M # subtract to avoid colliding with Temple blocks

        mirror_in_the_way = false
        for mirror in drop(eachrow(mirrors), 1) # first one is the lamp
            p = [mirror[1], mirror[2]]
            r = ray_segment_intersection(ray, (p, mirror_length, mirror[3]))
            if r[1] != 1 && r[2] < dist # 2 and 3 mean intersection closer than collision_point with temple
                mirror_in_the_way = true
                break
            end
        end

        if mirror_in_the_way
            # TODO: implement multiple reflections here
            continue
        end
        
        intersections = 0
        for ray2 in rays[1:end-1] # end-1 because we surely intersect with the last ray
            r = ray_ray_intersection(ray, ray2)
            if r != (4, 0, 0) && r != (2, 0, 0) # 2 and 4 mean no intersection
                if r[2] < dist # r[2] = t is the distance to the intersection
                    intersections += 1
                end
            end
        end

        collision_point = ray[1] + ray[2] * dist

        score_noise = 0 # rand() * 0.4 - 0.2 # seems to not help. And bigger noise is even worse
        score = dist - intersections * 1.2 + score_noise # penalty for every intersection, plus random noise
        if score > max_length
            max_length = score
            max_ray = (v, ray[2])
            max_endpoint = collision_point
        end
    end
    
    return (max_ray, max_length, max_endpoint)
end

function place_mirror(v::Array{Float64, 1}, e::Float64, rays, mirrors)
    p1 = [v[1] - cos(e) * 0.5, v[2] - sin(e) * 0.5] # leftmost possible point
    p2 = [v[1] + cos(e) * 0.5, v[2] + sin(e) * 0.5] # rightmost possible point
    p3 = [v[1] - cos(e) * 0.25, v[2] - sin(e) * 0.25] # point that puts center of mirror at v, 0.25 is the distance from the endpoint to the mirror edges

    tp1 = [floor(p1[1]), floor(p1[2])]
    tp2 = [floor(p2[1]), floor(p2[2])]
    
    blocks = [] # max 4 interesting blocks for our mirror
    
    for x in min(tp1[1],tp2[1]):max(tp1[1],tp2[1])
        for y in min(tp1[2],tp2[2]):max(tp1[2],tp2[2])
            block = block_from_point(temple, [x, y])
            if isnothing(block)
                continue
            end
            push!(blocks, block)
        end
    end

    if tp1 == tp2 && length(blocks) == 1
        return ([], false) # cannot place mirror anywhere, only one full block
    end

    for shift in -0.01:-0.05:-0.499 # try translating until we avoid all blocks
        p = [v[1] + cos(e) * shift, v[2] + sin(e) * shift]
        valid = true
        for block in blocks
            if segment_block_intersection((p, mirror_length, e), block)
                valid = false
                break
            end
        end
        if !valid
            continue
        end
        for ray in rays[1:end-2] # end-2 because we surely intersect with the last two rays - our mirror must touch them
            r = ray_segment_intersection(ray, (p, mirror_length, e))
            if r[1] != 1
                valid = false
                break
            end
        end
        if !valid
            continue
        end
        for mirror in drop(eachrow(mirrors), 1) # first one is the lamp
            if segment_segment_intersection((p, mirror_length, e), ([mirror[1], mirror[2]], mirror_length, mirror[3]))
                valid = false
                break
            end
        end
        if valid
            return (p, true)
        end
    end

    return ([], false) # couldn't place mirror anywhere
end

function generate_greedy_solution()
    rays = []

    ray, len, endpoint = generate_long_segment()
    push!(rays, ray)
    
    my_solution = Matrix{Float64}(undef, 0, 3)

    lamp_angle = atan(rays[1][2][2], rays[1][2][1])
    my_solution = vcat(my_solution, [rays[1][1][1] rays[1][1][2] lamp_angle])

    tinfo = @timed begin
    for i in 1:MIRRORS
        # calculate angle from endpoint to the starting point of the previous ray
        banned_angle = atan(ray[1][2] - endpoint[2], ray[1][1] - endpoint[1])
        ray, len, endpoint = longest_segment_from_point(endpoint, rays, my_solution, banned_angle)
        push!(rays, ray)

        end_point = rays[i + 1][1]
        direction1 = rays[i][2]
        direction2 = rays[i + 1][2]

        angle_between_rays = atan(direction2[2], direction2[1]) - atan(direction1[2], direction1[1])
        mirror_angle = atan(direction1[2], direction1[1]) + angle_between_rays / 2
        
        if mirror_angle < 0
            mirror_angle += 2π
        end
        
        mirror, okay = place_mirror(end_point, mirror_angle, rays, my_solution)
        if !okay
            println("Could not place mirror ", i)
            return [], []
        end
        my_solution = vcat(my_solution, [mirror[1] mirror[2] mirror_angle])
    end
    end # @time
    println("Time to greedily generate: ", tinfo.time, " s")

    return my_solution, rays
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

function fast_eval(rays)
    tinfo = @timed begin
    reset(temple)
    end
    println("Time to reset: ", tinfo.time, " s")
    for i in eachindex(rays[1:end-1])
        ray = rays[i]
        nxt_ray = rays[i + 1]
        draw_ray(ray[1][1], ray[1][2], nxt_ray[1][1], nxt_ray[1][2], 1, false)
    end
    dist = temple_ray_intersection(temple, rays[end])
    collision_point = rays[end][1] + rays[end][2] * dist
    draw_ray(rays[end][1][1], rays[end][1][2], collision_point[1], collision_point[2], 1, true) # draw last ray that collides with temple
    return fast_score()
end

# Random.seed!(0) # for reproducibility in Debug

sum_scores = 0.0
best_score = 0.0
num_scores = 0.0
num_valid_scores = 0.0

draw_temple(temple) # for fast_eval

while true
    global sum_scores, num_scores, num_valid_scores, best_score
    my_solution, rays = generate_greedy_solution()
    if isempty(my_solution)
        continue # was invalid and quit early
    end

    #length = calculate_solution_length(my_solution)
    #println("Length: ", length) # the correlation is sadly not super strong

    tinfo = @timed begin
    score = evaluate_solution(my_solution)
    end # @time
    println("Time to evaluate: ", tinfo.time, " s")

    tinfo = @timed begin
    fscore = fast_eval(rays) # for some reason this score is around 0.5% lower than the real score
    end # @time
    println("Fast score: ", fscore)
    println("Our time to evaluate: ", tinfo.time, " s")
    
    num_scores += 1

    if score > 0 # could use > 5 to remove solutions where the Ray immediately hits a wrong mirror and then a wall
        best_score = max(best_score, score)
        sum_scores += score
        num_valid_scores += 1
    end
    
    #println("Average score: ", sum_scores / num_valid_scores, " Percent valid scores: ", 100.0 * num_valid_scores / num_scores, " Best score: ", best_score)
    println("Average score: ", sum_scores / num_valid_scores, " Best score: ", best_score)
end