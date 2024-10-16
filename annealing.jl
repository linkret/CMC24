include("sve_kombinacije.jl")

function move_mirror_operation(solution::Matrix{Float64}, mirror_idx::Int64)::Vector{Matrix{Float64}}
    new_solutions = Vector{Matrix{Float64}}(undef, 0)
    step = 0.001 + rand() * 0.1 # TODO: decrease step with temperature/time
    dx = Float64[step, -step, 0., 0.]
    dy = Float64[0., 0., step, -step]

    is_first = (mirror_idx == 1)
    is_last = (mirror_idx == size(solution, 1)) # Maybe fix 8

    rays = trace_solution(solution)[1]
    nxt_ray = rays[mirror_idx + 1]
    prv_ray = is_first ? Ray(Point(), Direction()) : rays[mirror_idx - 1]
    nxt_nxt_ray = is_last ? Ray(Point(), Direction()) : rays[mirror_idx + 2]

    for i in 1:2, j in 1:2
        new_solution = copy(solution)
        new_solution[mirror_idx, 1] = solution[mirror_idx, 1] + dx[i]
        new_solution[mirror_idx, 2] = solution[mirror_idx, 2] + dy[j]
        
        # adjust current mirror
        if is_first
            collision_point = Point(new_solution[mirror_idx, 1:2])
            new_solution[mirror_idx, 3] = kut_izmedu_tocki(collision_point, nxt_ray.point)
        else
            case, t, u = ray_segment_intersection(prv_ray, Segment(Point(new_solution[mirror_idx, 1:2]), mirror_length, new_solution[mirror_idx, 3]))

            if case == 1
                continue
            end

            collision_point = prv_ray.point + t * prv_ray.direction
            angle_to_prv = kut_izmedu_tocki(collision_point, prv_ray.point)
            angle_to_nxt = kut_izmedu_tocki(collision_point, nxt_ray.point)
            angle_between_rays = (angle_to_prv + π - angle_to_nxt) % 2π
            mirror_angle = (angle_to_nxt + angle_between_rays / 2) % 2π
            
            # TODO: check
            mirror_point, success = place_mirror(collision_point, mirror_angle, Ray[], Matrix{Float64}(undef, 0, 3))
            
            if !success
                continue
            end

            new_solution[mirror_idx, 1] = mirror_point[1]
            new_solution[mirror_idx, 2] = mirror_point[2]
            new_solution[mirror_idx, 3] = mirror_angle
        end

        # adjust next mirror
        if !is_last
            angle_to_prv = kut_izmedu_tocki(nxt_ray.point, collision_point) # angle_to_nxt + pi
            angle_to_nxt = kut_izmedu_tocki(nxt_ray.point, nxt_nxt_ray.point)
            angle_between_rays = (angle_to_prv + π - angle_to_nxt) % 2π
            mirror_angle = (angle_to_nxt + angle_between_rays / 2) % 2π

            mirror_point, success = place_mirror(nxt_ray.point, mirror_angle, Ray[], Matrix{Float64}(undef, 0, 3))
            
            if !success
                continue
            end

            new_solution[mirror_idx + 1, 1] = mirror_point[1]
            new_solution[mirror_idx + 1, 2] = mirror_point[2]
            new_solution[mirror_idx + 1, 3] = mirror_angle
        end
        
        if !verify_solution(new_solution)
            # println("Verify failed")
            continue
        end

        push!(new_solutions, new_solution)
    end

    return new_solutions
end

function rotate_mirror_operation(solution::Matrix{Float64}, mirror_idx::Int64)::Vector{Matrix{Float64}}
    new_solutions = Vector{Matrix{Float64}}(undef, 0)
    step = 0.00001 + rand() * 0.0001 # TODO: decrease step with temperature/time

    is_first = (mirror_idx == 1)
    is_last = (mirror_idx == size(solution, 1)) # Maybe fix 8

    rays = trace_solution(solution)[1]
    nxt_ray = rays[mirror_idx + 1]
    prv_ray = is_first ? Ray(Point(), Direction()) : rays[mirror_idx - 1]
    nxt_nxt_ray = is_last ? Ray(Point(), Direction()) : rays[mirror_idx + 2]

    for dangle in [-step, step]
        new_solution = copy(solution)
        new_solution[mirror_idx, 3] = (new_solution[mirror_idx, 3] + dangle) % 2π

        # adjust current mirror
        if !is_first
            collision_point = Point(new_solution[mirror_idx, 1:2])
            mirror_point, success = place_mirror(collision_point, new_solution[mirror_idx, 3], Ray[], Matrix{Float64}(undef, 0, 3))

            if !success
                continue
            end

            new_solution[mirror_idx, 1] = mirror_point[1]
            new_solution[mirror_idx, 2] = mirror_point[2]
        end

        rays = trace_solution(new_solution)[1]

        # adjust next mirror
        if !is_last && length(rays) >= mirror_idx + 1
            prev_point = rays[mirror_idx].point
            cur_point = rays[mirror_idx + 1].point
            nxt_point = nxt_nxt_ray.point

            angle_to_prv = kut_izmedu_tocki(cur_point, prev_point)
            angle_to_nxt = kut_izmedu_tocki(cur_point, nxt_point)
            angle_between_rays = (angle_to_prv + π - angle_to_nxt) % 2π
            mirror_angle = (angle_to_nxt + angle_between_rays / 2) % 2π

            mirror_point, success = place_mirror(cur_point, mirror_angle, Ray[], Matrix{Float64}(undef, 0, 3))
            
            if !success
                continue
            end

            new_solution[mirror_idx + 1, 1] = mirror_point[1]
            new_solution[mirror_idx + 1, 2] = mirror_point[2]
            new_solution[mirror_idx + 1, 3] = mirror_angle
        end
        
        if !verify_solution(new_solution)
            # println("Verify failed")
            continue
        end

        push!(new_solutions, new_solution)
    end

    return new_solutions
end

function hill_climb(solution::Matrix{Float64})
    cur_solution = copy(solution)
    cur_score = evaluate_solution(cur_solution)
    best_solution = (cur_score, cur_solution)

    while true
        new_solutions = Vector{Matrix{Float64}}(undef, 0)
        for i in 1:size(cur_solution, 1)
            push!(new_solutions, move_mirror_operation(cur_solution, i)...)
            push!(new_solutions, rotate_mirror_operation(cur_solution, i)...)
        end

        best_new_solution = (0.0, Matrix{Float64}(undef, 0, 0))
        for new_solution in new_solutions
            new_score = evaluate_solution(new_solution)
            if new_score > best_new_solution[1]
                best_new_solution = (new_score, new_solution)
            end
        end

        if best_new_solution[1] > best_solution[1]
            best_solution = best_new_solution
            cur_solution = best_new_solution[2]
            cur_score = best_new_solution[1]
            println("New best score: ", cur_score)
        else
            println("WARNING: No better solution found")
        end
    end
end

# begin
#     cur_score = 0.0
#     start_solution = Matrix{Float64}(undef, 0, 0)
#     while score < 0.9
#         cur_score, start_solution = meet_in_the_middle()
#         println("Score: ", cur_score)
#     end
# end

start_solution = best_solution[2]

hill_climb(start_solution)