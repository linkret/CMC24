using LinearAlgebra
using Random
using Profile
using ProfileView

include("cmc24.jl")

function evaluate_and_draw(filename)
    # load the solution
    best_score, cmc24_solution = load_solution_file(filename)
    lamp, mirrors = load_solution(cmc24_solution, mirror_length)
    if !check_solution(temple, lamp, mirrors)
        return 0
    end
    
    # compute the ray path
    path = raytrace(temple, lamp, mirrors)
    
    # evaluate the solution
    total, vacant, score = evaluate(temple, path)
    # println(stderr, "Base plot has $(commas(vacant)) vacant of total $(commas(total)) pixels.")
    score_percent = 100. * score / vacant
    println(stderr, "Your CMC24 score is $(commas(score)) / $(commas(vacant)) = $(100. * score / vacant) %.")
    
    # create the presentation plot
    cmc24_plot(temple, lamp=lamp, mirrors=mirrors, path=path)

    return score_percent
end

evaluate_and_draw("solution.txt")

