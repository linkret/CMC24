# using Random
using Base.Iterators: drop
using Pkg
using Random
using JSON

include("cmc24.jl")
include("fast_eval.jl")

const PULL_OUT_SMALL = 0.1
const PULL_OUT_BIG = 2.0
const udaljenost_od_ruba=2.1

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
    return atan(dy,dx)# *(180 / π)  # convert from radians to degrees
end

function kut_izmedu_tocki(p1::IntPoint,p2::IntPoint)::Float64
    dy=p2.y-p1.y
    dx=p2.x-p1.x
    return atan(dy,dx)# *(180 / π)  # convert from radians to degrees
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

    step = 0.5
    razmatraniKutevi=RazmatraniKutevi[]

    for e in 0.0:step:359.9
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

najval=0
best=0

function searchFrom(v::IntPoint, depth::Int,
    banned_angle::Float64 = -1.0,
    banned_angle_range::Float64 = 0.1 )
    global najval
    if depth==9
        
        if fast_score()>najval
            najval=fast_score()
            println(najval)

            global best
            if best<najval
                best=najval
                draw_pixels_png()
            end
        end

        return
    end

    moji_susedi = susedi[v.x,v.y]
    should_break = false
    
    # moji_susedi=IntPoint[]
    # if depth >= 3
    #     moji_susedi = susedi[v.x,v.y]
    # else
    #     moji_susedi = susedi[v.x,v.y]
    #     moji_susedi = shuffle(moji_susedi)
    #     should_break = true
    # end
    
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
        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) )
        searchFrom(z, depth+1, a)
        draw_ray(Point(v.x/10,v.y/10), Point(z.x/10,z.y/10) , -1 )

        if should_break
            break
        end
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
                ok=false
                for pullOut in pocetak:korak:kraj
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
                            ok=true
                        end
                        prosli=length(privremeni_susedi[IntTocka.x,IntTocka.y])
                    end

                end

                if ok
                    push!(susedi[x,y], naj )
                    push!(susedi[x,200-y], IntPoint(naj.x, 200-naj.y) )
                    push!(susedi[200-x,y], IntPoint(200-naj.x, naj.y) )
                    push!(susedi[200-x,200-y], IntPoint(200-naj.x, 200-naj.y) )
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

const susedi = begin
    filename = "susedi.json"
    if isfile(filename)
        load_susedi_from_json(filename)
    else
        privremeni_susedi = Array{Vector{IntPoint}}(undef, 200, 200)
        stvori_susjede(privremeni_susedi)

        susedi = Array{Vector{IntPoint}}(undef, 200, 200)
        for x in 1:200
            for y in 1:200
                susedi[x,y]=IntPoint[]
            end
        end
        usavrsi_susjede(privremeni_susedi, susedi, PULL_OUT_SMALL, 0.1 , PULL_OUT_BIG)
    #    usavrsi_susjede(privremeni_susedi, susedi, PULL_OUT_BIG, -0.1 , PULL_OUT_SMALL)

        write_susedi_to_json(susedi, filename)
        susedi
    end
end


reset()
draw_temple(temple)



while true
    v = Point(0, 0)
    while point_in_temple(temple, Point(v.x, v.y)) || is_within_distance_of_boundary(Point(v.x,v.y))==false
        #x = rand(11:99)
        #y = rand(11:99)
        x=17
        y=17
        v=Point(x/10,y/10)
    end
    sleep(1)
    global najval
    global best
    najval=0
    println("Searching from $v")
    searchFrom(IntPoint(round( Int, v.x*10 ),round( Int, v.y*10 )), 0)
    println("najbolje do sad:")
    println(best)
end





