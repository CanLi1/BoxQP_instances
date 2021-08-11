function readproblem(filename)
    lines = readlines(filename)
    d = parse(Int, lines[1])
    c = [parse(Int, i) for i in split(lines[2])]
    Q = []
    for i in 1:d 
        new_row = [parse(Int, j) for j in split(lines[2+i])]
        push!(Q, new_row)
    end 

    return d, c, Q
end 

