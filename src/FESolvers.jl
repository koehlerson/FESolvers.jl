module FESolvers

import TimerOutputs
import ProgressMeter
using Ferrite
using LinearAlgebra

abstract type AbstractFEModel end
abstract type AbstractFEBuffer end
abstract type AbstractExportBuffer end
abstract type AbstractSolver end

####################################
######### General Interface ########
####################################
assemble!(::AbstractFEModel,::AbstractFEBuffer) = ()
apply_neumann!(::AbstractFEModel,::AbstractFEBuffer,t) = ()
converged_callback(::AbstractFEModel,::AbstractFEBuffer) = ()
export_callback(::AbstractFEModel,::AbstractFEBuffer, ::AbstractExportBuffer) = ()
print(::AbstractFEModel) = true

######### FEModel #########
getdirichletconstraints(::AbstractFEModel) = ()
build_febuffer(::AbstractFEModel) = ()
build_exportbuffer(::AbstractFEModel) = ()
gettimeinterval(::AbstractFEModel) = ()

######### FEBuffer #########
getstiffnessmatrix(buffer::AbstractFEBuffer) = ()
getrhs(buffer::AbstractFEBuffer) = ()
getcurrentsolution(buffer::AbstractFEBuffer) = ()

####################################
######## Quasi-Static Newton #######
####################################
struct QuasiStaticNewton{linsolve <: Function} <: AbstractSolver
    linearsolver::linsolve
    maxitr::Int
    restol::Float64
end

getincrement(::AbstractFEBuffer) = ()

function solve(model::AbstractFEModel,solver::QuasiStaticNewton)
    print_progress = print(model)
    print_progress ? TimerOutputs.enable_timer!(TimerOutputs.get_defaulttimer()) : TimerOutputs.disable_timer!(TimerOutputs.get_defaulttimer())
    TimerOutputs.reset_timer!()

    febuffer = build_febuffer(model)
    exportbuffer = build_exportbuffer(model) 

    print_progress && (p = ProgressMeter.ProgressUnknown("Newton solver step:"))
    timeinterval = gettimeinterval(model)
    
    K = getstiffnessmatrix(febuffer)
    f = getrhs(febuffer)
    u = getcurrentsolution(febuffer)
    Δu = getincrement(febuffer)
    dbcs = getdirichletconstraints(model)
 
    for t in timeinterval
        newton_itr = 0
        converged = false

        update!(dbcs, t)
        apply!(model, dbcs)
        apply_neumann!(model,febuffer,t)
        print_progress && (prog = ProgressMeter.ProgressThresh(globaldata.solver.restol, "Solving timestep $(t):", 1))
        while !converged 
            newton_itr += 1
            assemble!(model,febuffer)    
            apply_zero!(K,f,dbcs)
            norm_residual = norm(f[free_dofs(dbcs)])

            if norm_residual < solver.restol
                converged = true
            elseif newton_itr > solver.maxitr
                error("Newton without convergence")
            end

            Δu .= solver.linearsolver(K,f)
            u .-= Δu
            print_progress && ProgressMeter.update!(prog, norm_residual; showvalues=[(:iter, newton_itr), (:Δunorm, norm(Δu))])
        end
        converged_callback(model,febuffer)
        export_callback(model,febuffer)
        print_progress && ProgressMeter.next!(p, showvalues=[(:t, t),(:maxu, maximum(abs.(u)))])
 
    end
end

end
