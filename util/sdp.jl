include("read.jl")



using JuMP, Mosek, MosekTools, Gurobi, LinearAlgebra

function sdprelax(d, c, Q)
    m = Model(Mosek.Optimizer)

    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)

    @objective(m, Min, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    @SDconstraint(m, sdp, [1 transpose(x); x X] >= 0 )

    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1)
    status = optimize!(m)

    return m 
end 

function mcrelax(d, c, Q)
    m = Model(Mosek.Optimizer)

    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)

    @objective(m, Min, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))

    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1) 
    status = optimize!(m)

    return m 
end 

function boxqp(d, c, Q)
    m = Model(Gurobi.Optimizer)    
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)

    @objective(m, Min, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    status = optimize!(m)
    return m 
end 

function boxqp_bound(d, c, Q, bound)
    m = Model(Gurobi.Optimizer)    
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)
    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1) 
    @constraint(m, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d)>=bound)
    @constraint(m, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d) ==-1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    # @constraint(m, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d)>=bound)
    @objective(m, Min, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    status = optimize!(m)
    return m
end 

function boxqp_sdpcut(d, c, Q, dual_mat)
    m = Model(Gurobi.Optimizer)    
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)
    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1) 
    @constraint(m, dual_mat[1,1] + sum(x[i] * (dual_mat[i+1,1] + dual_mat[1,i+1]) for i in 1:d) + sum(dual_mat[i+1,j+1]*X[i,j] for i in 1:d for j in 1:d) >=0)
    @constraint(m, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d) ==-1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    @objective(m, Min, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    status = optimize!(m)
    return m
end 

function boxqp_sdpeigendecomp(d, c, Q, dual_mat)
    m = Model(Gurobi.Optimizer)    
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)
    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1) 
    s_mat = Symmetric(dual_mat)
    for i in 1:(d+1)
        s_mat[i,i] *= 2    
    end 
    s_mat = s_mat /2 
    eigvals, eig_vecs = eigen(s_mat) 
    for v_index in 1:(1+d)
        vec = eig_vecs[:, v_index]
        @constraint(m, vec[1] * vec[1] + sum(x[i] * (vec[i+1] * vec[1] *2) for i in 1:d) + sum(vec[i+1]*vec[j+1]*X[i,j] for i in 1:d for j in 1:d) >=0)
    end 
    @constraint(m, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d) ==-1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    @objective(m, Min, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    status = optimize!(m)
    return m
end 

function kelly_algo(d, c, Q)
    m = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, 0<=X[i in 1:d, j in 1:d], Symmetric)

    @objective(m, Min, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))

    #add McCormick 
    @constraint(m, mc1[i in 1:d, j in 1:d], X[i,j] <= x[i])
    @constraint(m, mc2[i in 1:d, j in 1:d], X[i,j] - x[i] - x[j] >= -1) 
    
    status = optimize!(m)
    sdp_X = [1 transpose(x); x X]
    eig_vals = eigvals(value.(sdp_X))
    eig_vecs = eigvecs(value.(sdp_X))
    round_of_cuts = 0 
    while eig_vals[1] < -1e-5        
        for v_index in 1:(d+1)
            if eig_vals[v_index] < 0
                v = eig_vecs[:, v_index]
                @constraint(m, sum(sdp_X[i,j] * v[i] * v[j] for i in 1:(d+1) for j in 1:(d+1))>=0)
            end 
        end 
        status = optimize!(m)
        eig_vals = eigvals(value.(sdp_X))
        eig_vecs = eigvecs(value.(sdp_X))        
        round_of_cuts += 1
    end 

    @constraint(m, -1/2*sum(Q[i][j] * X[i,j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d) ==-1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    @objective(m, Min, -1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))    
    status = optimize!(m)
    return m, round_of_cuts 
end 

function qcr(d, c, Q, dual_mat, bound)
    m = Model(Gurobi.Optimizer)    
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "Cuts", 0)
    set_optimizer_attribute(m, "Heuristics", 0)
    set_optimizer_attribute(m, "PreSolve", 0)
    @variable(m, 0<=x[i in 1:d]<=1)
    @variable(m, η)
    # @objective(m, Min, bound + dual_mat[1,1] + sum(x[i] * (dual_mat[i+1,1] + dual_mat[1,i+1]) for i in 1:d) + sum(dual_mat[i+1,j+1]* x[i] * x[j] for i in 1:d for j in 1:d))
    @constraint(m,  bound + dual_mat[1,1] + sum(x[i] * (dual_mat[i+1,1] + dual_mat[1,i+1]) for i in 1:d) + sum(dual_mat[i+1,j+1]* x[i] * x[j] for i in 1:d for j in 1:d) <= η)
    @constraint(m,   η >=-1/2*sum(Q[i][j] * x[i] * x[j] for i in 1:d for j in 1:d) - sum(c[i] * x[i] for i in 1:d))
    @objective(m, Min, η)
    status = optimize!(m)
    return m
end 
open("node.txt", "w") do io
    write(io, "instance boxqp c_cut sdp_cut sdp_cut_eigen kelly qcr\n")
end
files = ["spar020-100-1.in","spar020-100-2.in","spar020-100-3.in","spar030-060-1.in","spar030-060-2.in","spar030-060-3.in","spar030-070-1.in","spar030-070-2.in","spar030-070-3.in","spar030-080-1.in","spar030-080-2.in","spar030-080-3.in","spar030-090-1.in","spar030-090-2.in","spar030-090-3.in","spar030-100-1.in","spar030-100-2.in","spar030-100-3.in"]




for file in files
    filename = string("../basic/", file)

    d, c, Q = readproblem(filename)

    # mc_relax_model = mcrelax(d, c, Q)
    sdp_relax_model = sdprelax(d, c, Q)
    # boxqp_model = boxqp(d, c, Q)
    # boxqp_cutbound = boxqp_bound(d, c, Q, objective_value(sdp_relax_model))
    sdp_mat = dual(getindex(sdp_relax_model, :sdp))
    # boxqp_sdpcut_model = boxqp_sdpcut(d, c, Q, sdp_mat)
    boxqp_sdpeigendecomp_model = boxqp_sdpeigendecomp(d, c, Q, sdp_mat)
    # kelly_model, round_of_cuts = kelly_algo(d, c, Q)
    # qcr_model = qcr(d, c, Q, sdp_mat, objective_value(sdp_relax_model))
    open("node.txt", "a") do io
        write(io, string(file, " "))
        # write(io, string(Int(MOI.get(boxqp_model, MOI.NodeCount())), " "))
        # write(io, string(Int(MOI.get(boxqp_cutbound, MOI.NodeCount())), " "))
        # write(io, string(Int(MOI.get(boxqp_sdpcut_model, MOI.NodeCount())), " "))
        write(io, string(Int(MOI.get(boxqp_sdpeigendecomp_model, MOI.NodeCount())), " "))
        # write(io, string(Int(MOI.get(kelly_model, MOI.NodeCount())), " "))
        # write(io, string(Int(MOI.get(qcr_model, MOI.NodeCount())), " "))
        write(io, "\n")
    end

    
end 