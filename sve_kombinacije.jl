# using Random
using Base.Iterators: drop
using Pkg
using JSON

include("cmc24.jl")
include("fast_eval.jl")

const PULL_OUT_SMALL = 0.1
const PULL_OUT_BIG = 2.0

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

# skenira u krug i daje po svakom kutu najdaljeg susjeda(malo odmaknutog od zida doduse)
function potencijalne_susjede_tocke(v::Point)::Vector{RazmatraniKutevi}

    if point_in_temple(temple, v)
        return RazmatraniKutevi[]
    end

    step = 1
    razmatraniKutevi=RazmatraniKutevi[]

    for e in 0.0:step:359.9
        a = deg2rad(e)
        ray = Ray(v, Direction(cos(a), sin(a)))
        dist = temple_ray_intersection(temple, ray) # subtract to avoid colliding with Temple blocks

        if dist<8.1 #this sucks
            continue
        end

        endpoint = ray.point + ray.direction * (dist - 0.1)#(max(dist - PULL_OUT_M, 0.2))
        endpoint=Point(ceil(endpoint.x*10)/10,ceil(endpoint.y*10)/10)
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
    used_angles=[]
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


function stvori_susjede(susedi::Array{Vector{IntPoint}, 2})
    for x in 1:200
        for y in 1:200
            susedi[x,y]=IntPoint[]
        end
    end

    for x in 11:99#ide do 99 jer na preklapanjima se dogada krejziii stuf
        println(x)
        for y in 11:99
            
            """
            if x>25 && y>25 #ignorirajmo sve koji nisu na rubu jer budimo realni
                continue
            end
            """
            
            susedi[x, y] = smjerovi( Point(x/10,y/10) )
            susedi[x,200-y]=IntPoint[]
            susedi[200-x,y]=IntPoint[]
            susedi[200-x,200-y]=IntPoint[]
            for e in susedi[x,y]
                push!(susedi[x,200-y], IntPoint(e.x, 200-e.y) )
                push!(susedi[200-x,y], IntPoint(200-e.x, e.y) )
                push!(susedi[200-x,200-y], IntPoint(200-e.x, 200-e.y) )
            end
        end
    end
end

najval=0

function searchFrom(v::IntPoint, depth::Int,
    banned_angle::Float64 = -1.0,
    banned_angle_range::Float64 = 0.1 )
    global najval
    if depth==9
        
        if fast_score()>najval
            najval=fast_score()
            println(najval)
            draw_pixels_png()
        end

        return
    end

    moji_susedi=IntPoint[]

    if (depth >= 3)
        moji_susedi = susedi[v.x,v.y]
    else
        if length(susedi[v.x,v.y]) > 0
            push!(moji_susedi, rand(susedi[v.x,v.y]))
        end
    end
    
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
    end
end

function usavrsi_susjede(privremeni_susedi::Array{Vector{IntPoint}, 2}, susedi::Array{Vector{IntPoint}, 2})
    for x in 1:200
        for y in 1:200
            susedi[x,y]=IntPoint[]
        end
    end

    for x in 1:99
        for y in 1:99
            #susedi[x,y]=IntPoint[]
            #susedi[x,200-y]=IntPoint[]
            #susedi[200-x,y]=IntPoint[]
            #susedi[200-x,200-y]=IntPoint[]

            for sused in privremeni_susedi[x,y]
                a=kut_izmedu_tocki(IntPoint(x,y), sused )

                pocetna_tocka=Point(x/10,y/10)
                ray = Ray(pocetna_tocka, Direction(cos(a), sin(a)))
                dist = temple_ray_intersection(temple, ray)

                maxsus=0
                global naj=IntPoint(0,0)
                ok=false
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
                        if maxsus < length(privremeni_susedi[IntTocka.x,IntTocka.y])
                            maxsus=length(privremeni_susedi[IntTocka.x,IntTocka.y])
                            naj=IntTocka
                            ok=true
                        end
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
        usavrsi_susjede(privremeni_susedi, susedi)

        write_susedi_to_json(susedi, filename)
        susedi
    end
end

reset()
draw_temple(temple)

while true
    v = IntPoint(0, 0)
    while point_in_temple(temple, Point(v.x / 10, v.y / 10))
        x = rand(11:189)
        y = rand(11:189)
        v = IntPoint(x, y)
    end

    println("Searching from $v")
    searchFrom(v, 0)
end

"""
maxsus=0
println(susedi[20,20])
println( length(susedi[20,20]) )

suma=0
kol=0
reza=IntPoint(0,0)

for i in 1:99
    for j in 1:99
        global maxsus
        global suma
        global kol
        global reza
        if length(susedi[i,j] )>0
            suma+=length(susedi[i,j] )
            kol+=1
        end
        if maxsus <length(susedi[i,j] )
            reza=IntPoint(i,j)
            maxsus = length(susedi[i,j] )
        end
    end
end

println(susedi[33,93])

print("maximum:")
println(maxsus)
println(reza)
print("prosjek:")
println(suma/kol)
"""


