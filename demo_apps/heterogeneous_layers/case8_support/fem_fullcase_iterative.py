"""Higher-order full-deck FEM with an element-wise BDDC preconditioner.

The reference discretization is unchanged. A direct-solver comparison at
the same order verifies the iterative algebra before high-order refinement.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import ngsolve as ng


def make_space(mesh,order,names,vertex_coarse=False,far_order=None,roi=None,graded_order=None):
    if far_order is None:
        return ng.H1(mesh,order=order,dirichlet='|'.join(names),wb_withedges=not vertex_coarse),None
    if roi is None:raise ValueError('Variable order requires an explicit region of interest')
    roi=np.asarray(roi,float).reshape(2,3)
    space=ng.H1(mesh,order=far_order,dirichlet='|'.join(names),wb_withedges=not vertex_coarse,order_policy=ng.ORDER_POLICY.VARIABLE)
    points=np.array([v.point for v in mesh.vertices]);counts={}
    # HP refinement retains inactive parent edges/faces in mesh topology.
    # Enrich only entities of active volume cells; assigning modes to an
    # inactive parent would introduce zero-stiffness, unused unknowns.
    active_edges={};active_faces={};active_cells={}
    for element in mesh.Elements(ng.VOL):
        xyz=points[[v.nr for v in element.vertices]]
        if np.all(xyz.max(0)>=roi[0]) and np.all(xyz.min(0)<=roi[1]):
            local=min(order,graded_order) if graded_order is not None and element.type!=ng.ET.TET else order
            active_cells[element.nr]=local
            for edge in element.edges:active_edges[edge.nr]=max(local,active_edges.get(edge.nr,far_order))
            for face in element.faces:active_faces[face.nr]=max(local,active_faces.get(face.nr,far_order))
    for kind,indices in [(ng.EDGE,active_edges),(ng.FACE,active_faces),(ng.CELL,active_cells)]:
        for number,local in indices.items():space.SetOrder(ng.NodeId(kind,number),local)
        counts[str(kind)]=len(indices)
    # A full update also rebuilds coupling types and Dirichlet masks after
    # changing hierarchical entity orders. Updating only the dof numbering
    # leaves those arrays stale in variable-order spaces.
    space.Update();space.FinalizeUpdate()
    return space,dict(far_order=far_order,roi=roi.tolist(),enriched_entities=counts,graded_element_max_order=graded_order,
                     local_cell_order_counts={str(p):sum(q==p for q in active_cells.values()) for p in set(active_cells.values())})


def solve(case,mesh,order,bonus,threads,tol,maxiter,vertex_coarse=False,quad_audit=False,save_fields=None,far_order=None,roi=None,amg=False,targets=None,multigrid=False,mg_blocktype=None,graded_order=None):
    start=time.perf_counter();nt=len(case['conductors']);names=[f'terminal_{i}' for i in range(nt)]
    if amg:vertex_coarse=True
    targets=list(range(1,nt)) if targets is None else list(targets)
    if not targets or any(i<=0 or i>=nt for i in targets):raise ValueError('Invalid active terminal selection')
    space,order_audit=make_space(mesh,order,names,vertex_coarse,far_order,roi,graded_order);u,v=space.TnT()
    materials={f'eps_{r["epsilon_r"]:g}'.replace('.','p'):r['epsilon_r'] for r in case['plates']+case['coatings']}
    a=ng.BilinearForm(space,symmetric=True,condense=True);a+=mesh.MaterialCF(materials)*ng.grad(u)*ng.grad(v)*ng.dx(bonus_intorder=bonus)
    full=ng.BilinearForm(space,nonassemble=True);full+=mesh.MaterialCF(materials)*ng.grad(u)*ng.grad(v)*ng.dx(bonus_intorder=bonus)
    if multigrid:
        # Numeric type 4 groups each vertex with its incident edge modes,
        # then smooths face modes separately. This reduces high-p patch
        # storage without changing the conforming space or operator.
        pre=ng.Preconditioner(a,'multigrid',blocktype=['vertexpatch'] if mg_blocktype is None else mg_blocktype,inverse='sparsecholesky')
    else:
        pre=ng.Preconditioner(a,'bddc',coarsetype='h1amg') if amg else ng.Preconditioner(a,'bddc',inverse='sparsecholesky')
    print('start assembly',order,'dofs',space.ndof,flush=True);a.Assemble()
    print('assembled',order,'seconds',time.perf_counter()-start,flush=True)
    # Boundary lifts have sparse hierarchical coefficients. Retaining dense
    # lift and action vectors for every terminal would add several gigabytes
    # at this scale; only the solved potentials must remain resident.
    potentials=[];lifts=[];residual=[];iterations=[];free=np.array(space.FreeDofs(),bool);coupling=np.array(space.FreeDofs(coupling=True),bool)
    for i in targets:
        boundary=ng.GridFunction(space);boundary.Set(mesh.BoundaryCF({names[i]:1}),definedon=mesh.Boundaries('|'.join(names)))
        values=boundary.vec.FV().NumPy();indices=np.flatnonzero(values!=0)
        lifts.append((indices,values[indices].copy()))
    del boundary,values
    print('prepared boundary lifts',len(lifts),'seconds',time.perf_counter()-start,flush=True)
    C=np.zeros((len(targets),len(targets)));E=np.zeros_like(C)
    for column,i in enumerate(targets):
        print('start terminal',i,'seconds',time.perf_counter()-start,flush=True)
        g=ng.GridFunction(space);indices,values=lifts[column];g.vec.FV().NumPy()[indices]=values
        lift=g.vec.CreateVector();lift.data=g.vec;rhs=g.vec.CreateVector();rhs.data=a.mat*g.vec
        # Explicit projection prevents prescribed degrees from contaminating
        # the CG right-hand side or residual criterion.
        rhs.FV().NumPy()[~coupling]=0
        def progress(iteration,error):
            if iteration%10==0:print('CG',i,'iteration',iteration,'error',error,'seconds',time.perf_counter()-start,flush=True)
        inv=ng.CGSolver(a.mat,pre.mat,tol=tol,maxiter=maxiter,printrates=False,callback=progress)
        g.vec.data-=inv*rhs;g.vec.data+=a.harmonic_extension*g.vec
        action=g.vec.CreateVector();full.Apply(g.vec,action)
        fullrhs=g.vec.CreateVector();full.Apply(lift,fullrhs)
        error=float(np.linalg.norm(action.FV().NumPy()[free])/np.linalg.norm(fullrhs.FV().NumPy()[free]))
        residual.append(error);iterations.append(int(inv.iterations))
        if not np.isfinite(error) or error>1e-8:raise RuntimeError(f'Unconverged linear solve: residual {error}')
        av=action.FV().NumPy()
        C[:,column]=[float(np.dot(values,av[indices])) for indices,values in lifts]
        for row,other in enumerate([*potentials,g]):
            E[row,column]=E[column,row]=float(ng.InnerProduct(other.vec,action))
        potentials.append(g)
        print('solved terminal',i,'self_fF',float(ng.InnerProduct(lift,action))*8.8541878128e-12*case['length_unit_m']*1e15,
              'iterations',inv.iterations,'residual',error,'seconds',time.perf_counter()-start,flush=True)
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(E)):raise RuntimeError('Nonfinite terminal matrix')
    extra={}
    if quad_audit:
        qform=ng.BilinearForm(space,nonassemble=True);qform+=mesh.MaterialCF(materials)*ng.grad(u)*ng.grad(v)*ng.dx(bonus_intorder=bonus+4)
        Q=np.zeros_like(C);QR=np.zeros_like(C)
        for column,g in enumerate(potentials):
            q=g.vec.CreateVector();qform.Apply(g.vec,q)
            Q[:,column]=[float(ng.InnerProduct(other.vec,q)) for other in potentials]
            qv=q.FV().NumPy();QR[:,column]=[float(np.dot(values,qv[indices])) for indices,values in lifts]
        extra.update(quadrature_energy_C=Q.tolist(),quadrature_reaction_C=QR.tolist(),quadrature_mode='fixed converged fields; uncondensed operator evaluation with four additional integration orders',
            quadrature_relative_frobenius=float(np.linalg.norm(Q-C)/np.linalg.norm(C)))
    if save_fields:
        folder=Path(save_fields);folder.mkdir(parents=True,exist_ok=True)
        for i,g in zip(targets,potentials):g.Save(str(folder/f'terminal-{i}.sol'))
        extra['saved_fields']=str(folder)
    return dict(order=order,bonus_intorder=bonus,dofs=space.ndof,elements=mesh.ne,C=C.tolist(),
        C_fF=(C*8.8541878128e-12*case['length_unit_m']*1e15).tolist(),terminals=[case['conductors'][i]['id'] for i in targets],reference_terminal='GROUND',
        residual_max=max(residual),iterations=iterations,energy_reaction_discrepancy=float(np.linalg.norm(C-E)/np.linalg.norm(C)),
        reciprocity=float(np.linalg.norm(C-C.T)/np.linalg.norm(C)),
        material_volumes={m:float(ng.Integrate(1,mesh,definedon=mesh.Materials(m))) for m in materials},
        terminal_surface_areas=[float(ng.Integrate(1,mesh,definedon=mesh.Boundaries(n))) for n in names],
        seconds=time.perf_counter()-start,storage='Sparse exact boundary lifts and streamed reactions/energies; one dense solved potential per excitation',vertex_coarse=vertex_coarse,static_condensation=True,variable_order=order_audit,coarse_solver='h1amg' if amg else 'sparsecholesky',preconditioner=('multigrid-vertexpatch' if mg_blocktype is None else f'multigrid-blocktype-{mg_blocktype}') if multigrid else 'bddc',solver=f'NGSolve {ng.__version__}; conforming H1; static condensation; CG; uncondensed matrix-free residual/energy audit',**extra)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--mesh',required=True);p.add_argument('--output',required=True)
    p.add_argument('--orders',type=int,nargs='+',default=[2,3,4]);p.add_argument('--bonus',type=int,default=0);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--tol',type=float,default=1e-11);p.add_argument('--maxiter',type=int,default=1500)
    p.add_argument('--vertex-coarse',action='store_true')
    p.add_argument('--quad-audit',action='store_true');p.add_argument('--save-fields')
    far=p.add_mutually_exclusive_group();far.add_argument('--far-order',type=int)
    far.add_argument('--far-order-offset',type=int,help='Use far order p minus this offset for each requested p')
    p.add_argument('--roi',type=float,nargs=6)
    p.add_argument('--graded-order',type=int,help='Cap cell order on HP-generated prism, pyramid, and hex cells; shared traces retain the maximum incident order')
    p.add_argument('--amg',action='store_true');p.add_argument('--targets',type=int,nargs='+')
    p.add_argument('--multigrid',action='store_true')
    p.add_argument('--mg-blocktype',type=int,choices=[3,4,6,9])
    a=p.parse_args();ng.SetNumThreads(a.threads);ng.ngsglobals.msg_level=0
    path=Path(a.case);case=json.loads(path.read_text());mesh=ng.Mesh(a.mesh);out=Path(a.output);results=[]
    with ng.TaskManager():
        for order in a.orders:
            far_order=order-a.far_order_offset if a.far_order_offset is not None else a.far_order
            if far_order is not None and (far_order<1 or far_order>order):raise ValueError('Far order must lie between one and the requested local order')
            r=solve(case,mesh,order,a.bonus,a.threads,a.tol,a.maxiter,a.vertex_coarse,a.quad_audit and order==a.orders[-1],a.save_fields if order==a.orders[-1] else None,far_order,a.roi,a.amg,a.targets,a.multigrid,a.mg_blocktype,a.graded_order)
            r.update(case_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256(Path(a.mesh).read_bytes()).hexdigest(),mesh_file=Path(a.mesh).name)
            results.append(r);out.write_text(json.dumps(results,indent=2)+'\n')
            print('result',order,'self fF',np.diag(r['C_fF']).tolist(),flush=True)


if __name__=='__main__':main()
