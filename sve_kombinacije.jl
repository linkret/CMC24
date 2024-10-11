using Base.Iterators: drop
using Random
using JSON
using DataStructures: SortedDict

include("cmc24.jl")
include("fast_eval.jl")

const PULL_OUT_SMALL = 0.1
const PULL_OUT_BIG = 2.0
const udaljenost_od_ruba = 2.1

struct RazmatraniKutevi
    distance::Float64
    point::Point
    angle::Float64
end

struct IntPoint
    x::Int
    y::Int
end

function kut_izmedu_tocki(p1::Point,p2::Point)::Float64
    dy=p2.y-p1.y
    dx=p2.x-p1.x
    return atan(dy,dx)
end

function kut_izmedu_tocki(p1::IntPoint,p2::IntPoint)::Float64
    dy=p2.y-p1.y
    dx=p2.x-p1.x
    return atan(dy,dx)
end

function is_within_distance_of_boundary(point::Point, distance::Float64 = udaljenost_od_ruba)
    x=point.x
    y=point.y

    matrix_size=20
    # Calculate the boundaries (top, bottom, left, right)
    min_x, max_x = 0, matrix_size
    min_y, max_y = 0, matrix_size

    # Check if the point is within the specified distance from any boundary
    close_to_left   = abs(x - min_x) ≤ distance
    close_to_right  = abs(x - max_x) ≤ distance
    close_to_top    = abs(y - min_y) ≤ distance
    close_to_bottom = abs(y - max_y) ≤ distance

    # Return true if close to any boundary
    return close_to_left || close_to_right || close_to_top || close_to_bottom
end

# skenira u krug i daje po svakom kutu najdaljeg susjeda(malo odmaknutog od zida doduse)
function potencijalne_susjede_tocke(v::Point)::Vector{RazmatraniKutevi}

    if point_in_temple(temple, v)
        return RazmatraniKutevi[]
    end

    step = 0.01
    razmatraniKutevi=RazmatraniKutevi[]

    for e in 0.0:step:359.99999
        a = deg2rad(e)
        ray = Ray(v, Direction(cos(a), sin(a)))
        dist = temple_ray_intersection(temple, ray) # subtract to avoid colliding with Temple blocks

        if dist<9.1 #this sucks
            continue
        end

        endpoint = ray.point + ray.direction * (dist - 0.1)#(max(dist - PULL_OUT_M, 0.2))
        endpoint=Point(ceil(endpoint.x*10)/10,ceil(endpoint.y*10)/10)
        if is_within_distance_of_boundary(endpoint)==false
            continue
        end

        result=Point(0,0)
        kut=-69

        ok=false

        for i in 0:1:2
            for j in 0:1:2

                xx=endpoint.x-(i-1)/10
                yy=endpoint.y-(j-1)/10
                susjed=Point(xx,yy)
                if point_in_temple(temple, susjed)
                    continue
                end

                dist1=euclidean_distance(v, susjed)
                alfa=kut_izmedu_tocki(v,susjed)
                dist2 = temple_ray_intersection(temple, Ray(v, Direction(cos(alfa), sin(alfa))) )
                
                """
                println(dist1)
                println(dist2)
                println(v)
                println(susjed)
                println(rad2deg(alfa))
                println()
                """

                if(dist1<=dist2)#ako dist2 puca dalje od nase tocke onda imaju line of fire wooo
                    result=susjed
                    
                    kut=alfa
                    ok=true
                end

                if(ok)
                    break
                end
            end

            if(ok)
                break
            end

        end

        if (ok)
            push!( razmatraniKutevi, RazmatraniKutevi(dist, result, kut) )
        end 
    end

    return razmatraniKutevi#sort(razmatraniKutevi, by = dp -> dp.distance, rev=true)
end

"""
Default banned_angle_range is around 7 degrees (0.1 rad).
"""
function smjerovi(v::Point,banned_angle_range::Float64 = 0.1)::Vector{IntPoint}
    
    razmatraniKutevi=potencijalne_susjede_tocke(v)
    """
    println(  length(razmatraniKutevi))
    for g in 1:5
        println(razmatraniKutevi[g])
    end
    """
    #return Point[]

    susjedi=IntPoint[]
    used_angles=Float64[]
    used_angles=Float64[]
    #pretrazi 5 tocki i poslije pospajaj s
    while true
        naj=-5.0
        rjesenje=RazmatraniKutevi(0,Point(),0)
        for razmatranKut in razmatraniKutevi
            angle_visited=false
            a=razmatranKut.angle
            for banned_angle in used_angles
                if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                    angle_visited=true
                end
            end
            if angle_visited
                continue
            end

            if razmatranKut.distance>naj
                naj=razmatranKut.distance
                rjesenje=razmatranKut
            end

        end

        if naj<0.0
            break
        end
        
        push!(susjedi,IntPoint( round(Int, rjesenje.point.x*10) , round(Int, rjesenje.point.y*10) ) )
        push!(used_angles,rjesenje.angle)
    end

    """
    for p in susjedi
        println(p)
    end

    for kutek in used_angles
        println(rad2deg(kutek))
    end
    """

    return susjedi
end


function stvori_susjede(privremeni_susedi::Array{Vector{IntPoint}, 2})
    for x in 1:200
        for y in 1:200
            privremeni_susedi[x,y]=IntPoint[]
        end
    end

    for x in 11:99#ide do 99 jer na preklapanjima se dogada krejziii stuf
        println(x)
        for y in 11:99
            
            if x>udaljenost_od_ruba*10 && y>udaljenost_od_ruba*10 #ignorirajmo sve koji nisu na rubu jer budimo realni
                continue
            end
            
            privremeni_susedi[x, y] = smjerovi( Point(x/10,y/10) )
            
            for e in privremeni_susedi[x,y]
                push!(privremeni_susedi[x,200-y], IntPoint(e.x, 200-e.y) )
                push!(privremeni_susedi[200-x,y], IntPoint(200-e.x, e.y) )
                push!(privremeni_susedi[200-x,200-y], IntPoint(200-e.x, 200-e.y) )
            end
        end
    end
end

const najval = Ref{Float64}(0.)
const best = Ref{Float64}(0.)
const naj_solution = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 3))

function searchFrom(v::IntPoint, depth::Int,
    solution::Matrix{Float64} = Matrix{Float64}(undef, 0, 3),
    banned_angle::Float64 = -1.0,
    banned_angle_range::Float64 = 0.1 )
    global najval, best, naj_solution

    if depth==9
        if fast_score() > najval[]
            naj_solution[] = copy(solution)
            najval[] = fast_score()
            best[] = max(best[], najval[])
        end

        return
    end

    optimistic_score = fast_score() + (9 - depth) * 12
    if optimistic_score < najval[]
        return
    end

    moji_susedi = susedi[v.x,v.y]
    
    for z in moji_susedi
        a=kut_izmedu_tocki(v,z)+π
        b=a-π
        if banned_angle != -1.0
            if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                continue
            end
            if abs(b - banned_angle) < banned_angle_range || abs(b - banned_angle + 2π) < banned_angle_range || abs(b - banned_angle - 2π) < banned_angle_range
                continue
            end
        end

        # Place the mirror
        if depth == 0
            solution = vcat(solution, [v.x/10 v.y/10 b])
        else
            angle_between_rays = banned_angle + pi - b
            mirror_angle = b + angle_between_rays / 2
            mirror_angle = mirror_angle % 2π

            lp, lsuccess = place_mirror(Point(v.x/10, v.y/10), mirror_angle, Ray[], solution)
            if !lsuccess
                continue
            end
            solution = vcat(solution, [lp.x lp.y mirror_angle])
        end

        if !verify_solution(solution)
            solution = solution[1:end-1, :]
            continue
        end

        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) )
        searchFrom(z, depth+1, solution, a)
        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) , -1 )
        solution = solution[1:end-1, :]
    end
end

const max_solution_cnt = 100

function add_solution!(solutions::SortedDict{Float64, Matrix{Float64}}, score::Float64, solution::Matrix{Float64})
    solutions[score] = copy(solution)
    if length(solutions) > max_solution_cnt
        delete!(solutions, first(solutions).first)
    end
end

function searchFromHalf(v::IntPoint, depth::Int, sorted_solutions::SortedDict{Float64, Matrix{Float64}},
    solution::Matrix{Float64} = zeros(Float64, 1, 3),
    banned_angle::Float64 = -1.0,
    banned_angle_range::Float64 = 0.1 )

    if depth==4
        my_solution = vcat(solution, [v.x/10 v.y/10 banned_angle])
        #if verify_solution(my_solution, true)
        add_solution!(sorted_solutions, fast_score(), my_solution)
        #end
        return
    end
    
    for z in susedi[v.x, v.y]
        a=kut_izmedu_tocki(v,z)+π
        b=a-π
        if banned_angle != -1.0
            if abs(a - banned_angle) < banned_angle_range || abs(a - banned_angle + 2π) < banned_angle_range || abs(a - banned_angle - 2π) < banned_angle_range
                continue
            end
            if abs(b - banned_angle) < banned_angle_range || abs(b - banned_angle + 2π) < banned_angle_range || abs(b - banned_angle - 2π) < banned_angle_range
                continue
            end
        end

        # Place the mirror
        angle_between_rays = banned_angle + pi - b
        mirror_angle = b + angle_between_rays / 2
        mirror_angle = mirror_angle % 2π

        lp, lsuccess = place_mirror(Point(v.x/10, v.y/10), mirror_angle, Ray[], solution)
        if !lsuccess
            continue
        end

        solution = vcat(solution, [lp.x lp.y mirror_angle])

        if !verify_solution(solution)
            solution = solution[1:end-1, :]
            continue
        end

        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) )
        searchFromHalf(z, depth+1, sorted_solutions, solution, a)
        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) , -1 )
        solution = solution[1:end-1, :]
    end
end

function usavrsi_susjede(privremeni_susedi::Array{Vector{IntPoint}, 2}, susedi::Array{Vector{IntPoint}, 2} , pocetak::Float64,korak::Float64,kraj::Float64)

    for x in 1:99
        for y in 1:99

            for sused in privremeni_susedi[x,y]
                a=kut_izmedu_tocki(IntPoint(x,y), sused )

                pocetna_tocka=Point(x/10,y/10)
                ray = Ray(pocetna_tocka, Direction(cos(a), sin(a)))
                dist = temple_ray_intersection(temple, ray)

                prosli=0
                global naj=IntPoint(0,0)
                for pullOut in PULL_OUT_SMALL:0.1:PULL_OUT_BIG
                    endpoint = pocetna_tocka + ray.direction * (dist - pullOut)
                    tocka=Point(round(endpoint.x, digits=1) , round(endpoint.y, digits=1) )
                    if point_in_temple(temple, tocka)
                        continue
                    end

                    dist1=euclidean_distance(pocetna_tocka, tocka)
                    alfa=kut_izmedu_tocki(pocetna_tocka,tocka)
                    dist2 = temple_ray_intersection(temple, Ray(pocetna_tocka, Direction(cos(alfa), sin(alfa))) )

                    if(dist1<=dist2)#ako dist2 puca dalje od nase tocke onda imaju line of fire wooo
                        IntTocka=IntPoint(round(Int, tocka.x*10), round(Int, tocka.y*10))
                        if prosli < length(privremeni_susedi[IntTocka.x,IntTocka.y])
                            naj=IntTocka
                            push!(susedi[x,y], naj )
                        push!(susedi[x,200-y], IntPoint(naj.x, 200-naj.y) )
                        push!(susedi[200-x,y], IntPoint(200-naj.x, naj.y) )
                        push!(susedi[200-x,200-y], IntPoint(200-naj.x, 200-naj.y) )
                        end
                        prosli=length(privremeni_susedi[IntTocka.x,IntTocka.y])
                    end

                end

            end
        end
    end
end

function intpoint_to_dict(p::IntPoint)::Dict{String, Int}
    return Dict("x" => p.x, "y" => p.y)
end

function vector_to_list(v::Vector{IntPoint})::Vector{Dict{String, Int}}
    return [intpoint_to_dict(p) for p in v]
end

# Function to write the susedi matrix to a JSON file
function write_susedi_to_json(susedi::Array{Vector{IntPoint}}, filename::String)
    data = [[vector_to_list(susedi[i, j]) for j in 1:size(susedi, 2)] for i in 1:size(susedi, 1)]
    open(filename, "w") do file
        JSON.print(file, data)
    end
end

function dict_to_intpoint(d)::IntPoint
    return IntPoint(d["x"], d["y"])
end

function list_to_vector(l)::Vector{IntPoint}
    return [dict_to_intpoint(d) for d in l]
end

# Function to load the susedi matrix from a JSON file
function load_susedi_from_json(filename::String)::Array{Vector{IntPoint}}
    data = JSON.parsefile(filename)
    susedi = Array{Vector{IntPoint}}(undef, 200, 200)
    for i in 1:200
        for j in 1:200
            susedi[i, j] = list_to_vector(data[i][j])
        end
    end
    return susedi
end

# const precompute shifts for place_mirror

const start_shift = 0.00
const step_shift = -0.05
const end_shift = -0.50

const shifts = collect(start_shift:step_shift:end_shift)
const middle_shift = shifts[Int(ceil(length(shifts) / 2))]
const sorted_shifts = sort(shifts, by = x -> abs(x - middle_shift))

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

    for shift in sorted_shifts # try translating until we avoid all blocks
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

const susedi = begin
    sus_filename = "susedi.json"
    if isfile(sus_filename)
        load_susedi_from_json(sus_filename)
    else
        privremeni_susedi = Array{Vector{IntPoint}}(undef, 200, 200)
        tinfo = @timed begin
            stvori_susjede(privremeni_susedi)
        end
        println("Time to create susedi: ", tinfo.time, " s")

        susedi = Array{Vector{IntPoint}}(undef, 200, 200)
        for x in 1:200
            for y in 1:200
                susedi[x,y]=IntPoint[]
            end
        end
        usavrsi_susjede(privremeni_susedi, susedi, PULL_OUT_SMALL, 0.1 , PULL_OUT_BIG)
    #    usavrsi_susjede(privremeni_susedi, susedi, PULL_OUT_BIG, -0.1 , PULL_OUT_SMALL)

        write_susedi_to_json(susedi, sus_filename)
        susedi
    end
end

function get_mim_base()::Tuple{IntPoint, IntPoint, Float64}
    while true
        start1 = IntPoint(rand(11:99), rand(11:99))

        if point_in_temple(temple, Point(start1.x/10, start1.y/10))
            continue
        end

        if susedi[start1.x, start1.y] == IntPoint[]
            continue
        end

        start2 = rand(susedi[start1.x, start1.y])

        kut = kut_izmedu_tocki(start1, start2)

        have_options = false
        for sused in susedi[start1.x, start1.y]
            kut_sused = kut_izmedu_tocki(start1, sused)
            if abs(kut - kut_sused) > 0.1 # banned_angle_range
                have_options = true
                break
            end
        end

        if !have_options
            continue
        end

        for sused in susedi[start2.x, start2.y]
            kut_sused = kut_izmedu_tocki(start2, sused)
            if abs(kut + pi - kut_sused) > 0.1 # banned_angle_range
                return start1, start2, kut
            end
        end
    end
end

function verify_solution(solution::Matrix{Float64}, start_from_last::Bool = false)::Bool
    hit_mirrors = Set{Int}()
    hit_mirror = 0

    ray = Ray(Point(solution[1, 1], solution[1, 2]), Direction(cos(solution[1, 3]), sin(solution[1, 3])))

    itercnt = 0
    while itercnt < 20
        # check if ray can hit some mirror
        t_mirror = ∞
        for (m, mirror) ∈ enumerate(eachrow(solution[2:end, :]))
            (case, t, u) = ray_segment_intersection(ray, Segment(Point(mirror[1], mirror[2]), mirror_length, mirror[3]))
            if ((case == 2) || (case == 3)) && (t < t_mirror) && (t > ε)
                t_mirror = t
                push!(hit_mirrors, m + 1)
                hit_mirror = m + 1
            end
        end

        # check where ray would hit the temple
        t_temple = temple_ray_intersection(temple, ray)

        # closest hit point
        t = min(t_mirror, t_temple)
        hitting_point = ray.point + t * ray.direction

        # ray hit a mirror, calculate new direction
        if t_mirror < t_temple
            mirror_direction = Direction(cos(solution[hit_mirror, 3]), sin(solution[hit_mirror, 3]))
            normal = Direction(-mirror_direction[2], mirror_direction[1])

            ray = Ray(
                hitting_point,
                ray.direction - 2 * (ray.direction ⋅ normal) * normal
            )

            itercnt += 1
            continue
        end

        # ray hit the temple
        break
    end

    if itercnt >= 20
        return false
    end

    return length(hit_mirrors) >= size(solution, 1) - 1
end

function draw_rays(solution::Matrix{Float64}, color::Int, matrix::Matrix{UInt8})
    for (i, m1) in enumerate(eachrow(solution)[2:end-1])
        m2 = solution[i+2, :]
        draw_ray(Point(m1[1], m1[2]), Point(m2[1], m2[2]), color, i+2 == size(solution, 1), matrix)
    end
end

function meet_in_the_middle()
    global najval, naj_solution, best
    reset()

    left_solutions_dict = SortedDict{Float64, Matrix{Float64}}()
    right_solutions_dict = SortedDict{Float64, Matrix{Float64}}()

    start1, start2, start_alfa = get_mim_base()

    draw_ray(Point(start1.x/10, start1.y/10), Point(start2.x/10, start2.y/10))

    searchFromHalf(start1, 0, left_solutions_dict, [start2.x/10 start2.y/10 start_alfa + π], start_alfa)
    searchFromHalf(start2, 0, right_solutions_dict, [start1.x/10 start1.y/10 start_alfa], start_alfa + π)
    
    if isempty(left_solutions_dict) || isempty(right_solutions_dict)
        println("No solution found")
        return
    end

    println("Left solutions len: ", length(left_solutions_dict))
    println("Right solutions len: ", length(right_solutions_dict))

    best_solution = last(left_solutions_dict).second
    draw_rays(best_solution, 1, pixels)
    best_solution = best_solution[end:-1:2, :]
    najval[] = 0
    searchFrom(start2, 5, best_solution, start_alfa + π)
    if najval[] == 0 # try the other starting point, why not
        reset()
        best_solution = last(right_solutions_dict).second
        draw_rays(best_solution, 1, pixels)
        best_solution = best_solution[end:-1:2, :]
        searchFrom(start1, 5, best_solution, start_alfa)
    end
    best_solution = naj_solution[]
    best_score = najval[]

    reset()
    draw_ray(Point(start1.x/10, start1.y/10), Point(start2.x/10, start2.y/10))

    println("Starting score: ", best_score)

    left_solutions = reverse(collect(values(left_solutions_dict))) # reverse to start from best solutions
    right_solutions = reverse(collect(values(right_solutions_dict)))

    left_matrices = [copy(pixels) for i in left_solutions]
    right_matrices = [copy(pixels) for i in right_solutions]

    for (idx, left_solution) in enumerate(left_solutions)
        draw_rays(left_solution, 1, left_matrices[idx])
    end

    for (idx, right_solution) in enumerate(right_solutions)
        draw_rays(right_solution, 1, right_matrices[idx])
    end

    for (lidx, left_solution) in enumerate(left_solutions)
        left_matrix = left_matrices[lidx]
        
        for (ridx, right_solution) in enumerate(right_solutions)
            right_matrix = right_matrices[ridx]

            mirrors_intersection = false
            for left_mirror in eachrow(left_solution[2:end-1, :])
                for right_mirror in eachrow(right_solution[2:end-1, :])
                    if segment_segment_intersection(
                        Segment(Point(left_mirror[1], left_mirror[2]), mirror_length, left_mirror[3]),
                        Segment(Point(right_mirror[1], right_mirror[2]), mirror_length, right_mirror[3]))
                        mirrors_intersection = true
                        break
                    end
                end
            end

            if mirrors_intersection
                continue
            end

            solution = vcat(left_solution[2:end-1, :], right_solution[2:end, :]) # TODO: funky
            solution = solution[end:-1:1, :] # reverse rows

            if !verify_solution(solution)
                continue
            end

            score = slow_score(left_matrix, right_matrix)
            if score > best_score
                best_score = score
                best_solution = solution
                println("New best score: ", best_score)
            end
        end
    end

    println("Best solution: ")
    println(best_score)
    for m in eachrow(best_solution)
        println("$(m[1]) $(m[2]) $(m[3])")
    end

    return evaluate_solution(best_solution)
end

reset()
draw_temple(temple)

while false
    tinfo = @timed begin
        meet_in_the_middle()
    end
    println("Time to meet in the middle: ", tinfo.time, " s")
end

while true
    v = Point(0, 0)
    while point_in_temple(temple, Point(v.x, v.y)) || is_within_distance_of_boundary(Point(v.x,v.y))==false
        x = rand(11:99)
        y = rand(11:99)
        # x = 17
        # y = 17
        v=Point(x/10, y/10)
    end
    global najval, best
    najval[] = 0
    println("Searching from $v")
    searchFrom(IntPoint(round( Int, v.x*10 ),round( Int, v.y*10 )), 0)
    evaluate_solution(naj_solution[])
    println("najbolje do sad: ", best[])
    break
end





