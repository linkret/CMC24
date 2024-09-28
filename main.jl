# using Random
using Base.Iterators: drop

include("cmc24.jl")
include("fast_eval.jl")

const MIRRORS = 8
const PULL_OUT_L = 0.5
const PULL_OUT_M = 0.5 # 1.2 was probably too extreme

function generate_segment()::Tuple{Ray, Float64, Point}
    v = Point(0., 0.)
    while point_in_temple(temple, v)
        x = rand() * 19.0 + 1
        y = rand() * 19.0 + 1
        v = Point(x, y)
    end
    
    # Randomly select an initial angle as e = [cos(α), sin(α)]
    angle = rand() * 2 * π
    e = Point(cos(angle), sin(angle))
    
    ray = Ray(v, e)
    
    # this method call is a bit slow
    dist = temple_ray_intersection(temple, ray) - PULL_OUT_L # subtract to avoid colliding Lamp with Temple blocks
    collision_point = ray.point + ray.direction * dist

    rev_ray = Ray(v, -1 * e)
    rev_dist = temple_ray_intersection(temple, rev_ray) - PULL_OUT_M # subtract to avoid colliding Mirror with Temple blocks
    rev_collision_point = rev_ray.point + rev_ray.direction * rev_dist
    
    return Ray(collision_point, -1 * e), dist + rev_dist, rev_collision_point
end

function generate_long_segment(iter_cnt::Int = 100)::Tuple{Ray, Float64, Point}
    ray, length, endpoint = generate_segment()
    for i in 1:iter_cnt
        new_ray, new_length, new_endpoint = generate_segment()
        if new_length > length
            ray = new_ray
            length = new_length
            endpoint = new_endpoint
        end
    end
    return ray, length, endpoint
end

"""
    longest_segment_from_point(v::Array{Float64, 1}, rays, banned_angle::Float64 = -1.0, banned_angle_range::Float64 = 0.20)

    Returns ((v, e), length, endpoint) where v is the starting point, e is the direction, length is the length of the ray, and endpoint is the point of collision.
    The ray is generated from the point v, and the angle of the ray is chosen to maximize the length of the ray.
    The arguemnt rays is used to avoid generating rays that are too close to the previous rays.
    The angle of the ray is chosen from the range [0, 2π) excluding the banned_angle ± banned_angle_range.
    Default banned_angle_range is around 7 degrees (0.1 rad).
"""
function longest_segment_from_point(
    v::Point,
    rays::Array{Ray},
    mirrors::Matrix{Float64},
    banned_angle::Float64 = -1.0,
    banned_angle_range::Float64 = 0.1
)::Tuple{Vector{Ray}, Float64, Point}
    max_length = -1000.0
    max_rays = [Ray(Point(0., 0.), Direction(0., 0.))]
    max_endpoint = Point(0., 0.)

    if point_in_temple(temple, v)
        println("POINT IN TEMPLE BRO $v") # TODO: diagnose and fix
        return (Ray[], max_length, max_endpoint)
    end

    # TODO: can make a circular interval radian sweep, where each 1m segment belonging to a block is converted to an interval of angles with a distance
    # Then, we compute intervals of the closest-segments so we know it for each and every angle
    # We can use this to avoid any calls to temple_ray_intersection, which seems to be the bottleneck
    # This could allow us to try even more angles (finer angle-step), or more starting-points

    is_last_ray = (length(rays) == MIRRORS)
    draw = (length(rays) >= MIRRORS - 0)
    step = is_last_ray ? 0.5 : 1.0 # if we are generating the last ray, we can use a smaller step
    
    for e in 0.0:step:359.9
        a = deg2rad(e)
        
        if banned_angle != -1.0
            if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                continue
            end
        end
        
        ray = Ray(v, Direction(cos(a), sin(a)))
        new_rays = [deepcopy(ray)]
        dist = 0.0
        sum_dist = 0.0
        
        while length(new_rays) < 20
            dist = temple_ray_intersection(temple, ray) # subtract to avoid colliding with Temple blocks
            collision_dist = 0.0

            mirror_in_the_way = false
            new_ray = Ray(Point(0., 0.), Point(0., 0.))

            for mirror in drop(eachrow(mirrors), 1) # first one is the lamp
                p = Point(mirror[1], mirror[2])
                r = ray_segment_intersection(ray, Segment(p, mirror_length, mirror[3]))
                if r[1] != 1 && r[2] < dist # 2 and 3 mean intersection
                    mirror_in_the_way = true
                    n = mirror[3] - π / 2 # normal angle
                    alpha = 2 * n - atan(ray.direction[2], ray.direction[1]) # angle of reflection
                    collision_dist = r[2]
                    collision_point = ray.point + ray.direction * r[2]
                    new_ray = Ray(collision_point, -1 * Point(cos(alpha), sin(alpha)))
                    break
                end
            end

            if mirror_in_the_way
                push!(new_rays, deepcopy(new_ray))
                ray = deepcopy(new_ray)
                sum_dist += collision_dist
            else
                sum_dist += dist
                break
            end
        end

        if length(new_rays) >= 20 # to break infinite reflection loops
            continue
        end

        if sum_dist < 7.0 && max_length > 7.0 # don't even waste time evaluating this angle
            continue
        end
        
        intersections = 0
        for ray2 in rays[1:end-1] # end-1 because we surely intersect with the last ray
            r = ray_ray_intersection(ray, ray2)
            if r[1] != 4 && r[1] != 2 # 2 and 4 mean no intersection
                if r[2] < dist # r[2] = t is the distance to the intersection
                    # TODO: should use real distances - it's easily possible that ray2 ends before this collision point, and we make a mistake
                    intersections += 1
                end
            end
        end

        score = sum_dist - intersections * 1.2 - (length(new_rays) - 1) * 2.0 # applies penalties

        endpoint = ray.point + ray.direction * (max(dist - PULL_OUT_M, 0.2)) # TODO: inspect

        if draw
            for i in eachindex(new_rays[1:end-1])
                draw_rectangle_around_line(new_rays[i].point, new_rays[i + 1].point, 1)
            end
            draw_rectangle_around_line(ray.point, endpoint, 1)
            score = fast_score()
        end

        if score > max_length
            max_length = score
            max_rays = deepcopy(new_rays)
            max_endpoint = deepcopy(endpoint)
        end

        if draw
            for i in eachindex(new_rays[1:end-1])
                draw_rectangle_around_line(new_rays[i].point, new_rays[i + 1].point, -1)
            end
            draw_rectangle_around_line(ray.point, endpoint, -1)
        end
    end
    
    return (max_rays, max_length, max_endpoint)
end

function place_mirror(v::Point, e::Float64, rays::Array{Ray}, mirrors::Matrix{Float64})::Tuple{Point, Bool}
    p1 = Point(v[1] - cos(e) * 0.5, v[2] - sin(e) * 0.5) # leftmost possible point
    p2 = Point(v[1] + cos(e) * 0.5, v[2] + sin(e) * 0.5) # rightmost possible point

    tp1 = [floor(p1[1]), floor(p1[2])]
    tp2 = [floor(p2[1]), floor(p2[2])]
    
    blocks = Vector{Block}(undef, 0) # will have max 4 interesting blocks for our mirror
    
    for x in min(tp1[1],tp2[1]):max(tp1[1],tp2[1])
        for y in min(tp1[2],tp2[2]):max(tp1[2],tp2[2])
            block = block_from_point(temple, Point(x, y))
            if isnothing(block)
                continue
            end
            push!(blocks, block)
        end
    end

    if tp1 == tp2 && length(blocks) == 1
        return (Point(0., 0.), false) # cannot place mirror anywhere, only one full block
    end

    for shift in -0.01:-0.05:-0.499 # try translating until we avoid all blocks
        p = Point(v[1] + cos(e) * shift, v[2] + sin(e) * shift)
        valid = true
        for block in blocks
            if segment_block_intersection(Segment(p, mirror_length, e), block)
                valid = false
                break
            end
        end
        if !valid
            continue
        end
        for ray in rays[1:end-2] # end-2 because we surely intersect with the last two rays - our mirror must touch them
            r = ray_segment_intersection(ray, Segment(p, mirror_length, e))
            if r[1] != 1
                valid = false
                break
            end
        end
        if !valid
            continue
        end
        for mirror in drop(eachrow(mirrors), 1) # first one is the lamp
            if segment_segment_intersection(Segment(p, mirror_length, e), Segment(Point(mirror[1], mirror[2]), mirror_length, mirror[3]))
                valid = false
                break
            end
        end
        if valid
            return (p, true)
        end
    end

    return (Point(0., 0.), false) # couldn't place mirror anywhere
end

function generate_greedy_solution(best_score::Float64 = 0.0)::Tuple{Matrix{Float64}, Array{Ray}}
    rays = Array{Ray}(undef, 0)

    ray, len, endpoint = generate_long_segment()
    push!(rays, deepcopy(ray))
    draw_ray(ray.point, endpoint, 1, true)
    
    my_solution = Matrix{Float64}(undef, 0, 3) # TODO: maybe use struct Mirror

    lamp_angle = atan(rays[1].direction[2], rays[1].direction[1])
    my_solution = vcat(my_solution, [rays[1].point[1] rays[1].point[2] lamp_angle])

    solution_has_multiple_reflexions = false

    max_heuristic = 11.0 # how many % of area a single ray (mirror) covers in the best possible case

    tinfo = @timed begin
    for i in 1:MIRRORS
        if fast_score() + (MIRRORS + 1 - i) * max_heuristic < best_score # heuristic to quit early
            println("Gave up at $i, score was only $(fast_score())")
            return Matrix{Float64}(undef, 0, 0), Ray[]
        end
        direction1 = ray.direction # if we had multi-reflections last iteration, this will be the last ray's angle

        banned_angle = atan(ray.point[2] - endpoint[2], ray.point[1] - endpoint[1])
        new_rays, len, endpoint = longest_segment_from_point(endpoint, rays, my_solution, banned_angle)
        if length(new_rays) == 0 # There was an error
            return Matrix{Float64}(undef, 0, 0), Ray[]
        end
        ray = new_rays[1]
        push!(rays, deepcopy(ray))
        
        for j in eachindex(new_rays[1:end-1])
            solution_has_multiple_reflexions = true
            draw_ray(new_rays[j].point, new_rays[j + 1].point, 1, false) # can probably just be rectangles
        end
        draw_ray(new_rays[end].point, endpoint, 1, true)
        
        # calculate angle from endpoint to the starting point of the previous ray
        mirror_point = rays[i + 1].point
        direction2 = rays[i + 1].direction

        # if solution_has_multiple_reflexions
        #     println("Step $i")
        #     println(new_rays)
        #     println("mirror_point: ", mirror_point)
        #     println("enpoint:", endpoint)
        # end

        angle_between_rays = atan(direction2[2], direction2[1]) - atan(direction1[2], direction1[1])
        mirror_angle = atan(direction1[2], direction1[1]) + angle_between_rays / 2
        
        if mirror_angle < 0
            mirror_angle += 2π
        end
        
        mirror, okay = place_mirror(mirror_point, mirror_angle, rays, my_solution) # TODO: missing new_rays checks here
        if !okay
            println("Could not place mirror ", i)
            return Matrix{Float64}(undef, 0, 0), Ray[]
        end
        my_solution = vcat(my_solution, [mirror[1] mirror[2] mirror_angle])
        ray = new_rays[end]
    end
    end # @time
    println("Time to greedily generate: ", tinfo.time, " s")

    # if solution_has_multiple_reflexions == false
    #     println("No multiple reflexions!")
    #     return Matrix{Float64}(undef, 0, 0), Ray[]
    # end

    return my_solution, rays
end

function calculate_solution_length(solution::Matrix{Float64})::Float64
    sum = 0
    for i in 1:MIRRORS
        cur_point = solution[i, 1:2]
        nxt_point = solution[i + 1, 1:2]
        dist = norm(nxt_point - cur_point)
        sum += dist
    end
    return sum
end

# TODO: this method doesn't work anymore, it should call ray.point and ray.direction instead of ray[1] and ray[2]
# function fast_eval(rays)::Float64
#     reset()
#     for i in eachindex(rays[1:end-1])
#         ray = rays[i]
#         nxt_ray = rays[i + 1]
#         draw_ray(ray[1][1], ray[1][2], nxt_ray[1][1], nxt_ray[1][2], 1, false)
#     end
#     dist = temple_ray_intersection(temple, rays[end])
#     collision_point = rays[end][1] + rays[end][2] * dist
#     draw_ray(rays[end][1][1], rays[end][1][2], collision_point[1], collision_point[2], 1, true) # draw last ray that collides with temple
#     return fast_score()
# end

function main()
    # Random.seed!(0) # for reproducibility in Debug

    sum_scores = 0.0
    my_best_score = 0.0
    num_scores = 0.0
    num_valid_scores = 0.0

    draw_temple(temple) # for fast_eval.jl

    while true
        reset() # for fast_eval.jl

        my_solution, rays = generate_greedy_solution(my_best_score)
        if isempty(my_solution)
            continue # solution was invalid and quit early
        end

        score = fast_score() # not 100% accurate
        #tinfo = @timed begin
        #score = evaluate_solution(my_solution) # 100% accurate
        #end
        #println("Time to evaluate: ", tinfo.time, " s")
        println("Score: ", score)
        
        num_scores += 1

        if score > 0
            if score > best_score[1]
                score = evaluate_solution(my_solution) # 100% accurate
            end
            my_best_score = max(my_best_score, score)
            sum_scores += score
            num_valid_scores += 1
        end

        # break # TODO: remove
        
        #println("Average score: ", sum_scores / num_valid_scores, " Percent valid scores: ", 100.0 * num_valid_scores / num_scores, " Best score: ", best_score)
        println("Average score: ", sum_scores / num_valid_scores, " Best score: ", my_best_score)
    end
end

main()