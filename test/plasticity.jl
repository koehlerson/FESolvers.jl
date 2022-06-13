using FESolvers, Ferrite, Tensors, SparseArrays, LinearAlgebra
include("definitions-plasticity.jl")

struct PlasticityModel <: AbstractFEModel
    grid
    interpolation
    dh
    material
    traction_magnitude
    timeinterval
    dbcs
end

function PlasticityModel()
    E = 200.0e9
    H = E/20
    ν = 0.3 
    σ₀ = 200e6
    material = J2Plasticity(E, ν, σ₀, H)

    L = 10.0
    w = 1.0
    h = 1.0
    n_timesteps = 10
    u_max = zeros(n_timesteps)
    traction_magnitude = 1.e7 * range(0.5, 1.0, length=n_timesteps)

    n = 2
    nels = (10n, n, 2n)
    P1 = Vec((0.0, 0.0, 0.0))
    P2 = Vec((L, w, h))
    grid = generate_grid(Tetrahedron, nels, P1, P2)
    interpolation = Lagrange{3, RefTetrahedron, 1}()
    dh = create_dofhandler(grid, interpolation)
    dbcs = create_bc(dh, grid)
    return PlasticityModel(grid,interpolation,dh,material,traction_magnitude,1:n_timesteps,dbcs)
end

struct PlasticityFEBuffer <: AbstractFEBuffer
    cellvalues
    facevalues
    K
    r
    u
    states
    states_old
end

function FESolvers.build_febuffer(model::PlasticityModel)
    dh = model.dh
    n_dofs = ndofs(dh)
    cellvalues, facevalues = create_values(model.interpolation)
    u  = zeros(n_dofs)
    Δu = zeros(n_dofs)
    r = zeros(n_dofs)
    K = create_sparsity_pattern(dh)
    nqp = getnquadpoints(cellvalues)
    states = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.grid)]
    states_old = [J2PlasticityMaterialState() for _ in 1:nqp, _ in 1:getncells(model.grid)]
    return PlasticityFEBuffer(cellvalues,facevalues,K,r,u,states,states_old)
end

function FESolvers.apply_neumann!(model::PlasticityModel,buffer::PlasticityFEBuffer,t)
    nu = getnbasefunctions(buffer.cellvalues)
    re = zeros(nu)
    facevalues = buffer.facevalues
    grid = model.grid
    traction = Vec((0.0, 0.0, model.traction_magnitude[t]))

    for (i, cell) in enumerate(CellIterator(model.dh))
        fill!(re, 0)
        eldofs = celldofs(cell)
        for face in 1:nfaces(cell)
            if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "right")
                reinit!(facevalues, cell, face)
                for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
                    for i in 1:nu
                        δu = shape_value(facevalues, q_point, i)
                        re[i] -= (δu ⋅ traction) * dΓ
                    end
                end
            end
        end
        buffer.r[eldofs] .+= re
    end   
end

FESolvers.assemble!(model::PlasticityModel,buffer::PlasticityFEBuffer) = doassemble!(buffer.cellvalues,buffer.facevalues,buffer.K, buffer.r,
                                                                          model.grid,model.dh,model.material,buffer.u,buffer.states,buffer.states_old)
FESolvers.converged_callback(model::PlasticityModel,buffer::PlasticityFEBuffer) = buffer.states_old .= buffer.states
function FESolvers.export_callback(model::PlasticityModel,febuffer::PlasticityFEBuffer, exportbuffer)
    push!(exportbuffer,maximum(abs.(febuffer.u)))
end
FESolvers.print(::PlasticityModel) = true

######### FEModel #########
FESolvers.getdirichletconstraints(model::PlasticityModel) = model.dbcs
FESolvers.build_exportbuffer(model::PlasticityModel) = Float64[]
FESolvers.gettimeinterval(model::PlasticityModel) = model.timeinterval

######### FEBuffer #########
FESolvers.getstiffnessmatrix(buffer::AbstractFEBuffer) = buffer.K
FESolvers.getrhs(buffer::AbstractFEBuffer) = buffer.r
FESolvers.getcurrentsolution(buffer::AbstractFEBuffer) = buffer.u

model = PlasticityModel()
solver = QuasiStaticNewton(\,8,1.0)

exportbuffer = solve(model,solver)
