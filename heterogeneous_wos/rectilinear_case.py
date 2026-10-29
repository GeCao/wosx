"""Exact coordinate partition and weak charges for interleaved box unions.

The grid is a geometric arrangement, not a voxel approximation or a PDE solve.
Every original metal/material face is retained. Piecewise trilinear lifts give
bilinear face measures because their cellwise Laplacian vanishes identically.
"""
from pathlib import Path
import json
import numpy as np


def _rectangles(mask):
    """Greedily merge equal nonzero cells; preserve every material label."""
    a = mask.copy()
    for i in range(a.shape[0]):
        j = 0
        while j < a.shape[1]:
            label = a[i, j]
            if not label:
                j += 1
                continue
            end = j+1
            while end < a.shape[1] and a[i, end] == label:
                end += 1
            stop = i+1
            while stop < a.shape[0] and np.all(a[stop, j:end] == label):
                stop += 1
            a[i:stop, j:end] = 0
            yield i, stop, j, end, label
            j = end


class RectilinearCase:
    def __init__(self, metadata, padding=.3):
        self.metadata = metadata
        if metadata.get('schema_version') != 6 or metadata.get('reference') != 'insulating_window':
            raise ValueError('Schema 6 with insulating window required')
        self.outer = np.asarray(metadata['outer_box'], float)
        self.boxes = np.concatenate([c['boxes'] for c in metadata['conductors']])
        self.tags = np.concatenate([np.full(len(c['boxes']), i, np.int32)
                                    for i, c in enumerate(metadata['conductors'])])
        self.nt = len(metadata['conductors'])
        self.padding = float(padding)
        if padding <= 0 or not np.isfinite(padding):
            raise ValueError('Positive finite lift padding required')
        records = metadata['plates']+metadata['coatings']
        all_bounds = np.concatenate([self.boxes, [r['bounds'] for r in records], self.outer[None]])
        if not np.isfinite(all_bounds).all() or np.any(all_bounds[:, 0] >= all_bounds[:, 1]):
            raise ValueError('Finite positive-volume boxes required')
        if np.any(all_bounds[:, 0] < self.outer[0]-1e-10) or np.any(all_bounds[:, 1] > self.outer[1]+1e-10):
            raise ValueError('Geometry outside window')
        for i, b in enumerate(self.boxes):
            other = self.boxes[self.tags != self.tags[i]]
            gap = np.maximum(np.maximum(b[0]-other[:, 1], other[:, 0]-b[1]), 0)
            if np.any(np.linalg.norm(gap, axis=1) < 1e-11):
                raise ValueError('Different terminals touch or overlap')
        # Extra planes limit lift support outside each net's bounding box.
        extra = []
        for c in metadata['conductors']:
            b = np.asarray(c['boxes'])
            extra.extend([np.maximum(b[:, 0].min(0)-padding, self.outer[0]),
                          np.minimum(b[:, 1].max(0)+padding, self.outer[1])])
        extra = np.asarray(extra)
        self.coordinates = [np.unique(np.round(np.r_[all_bounds[:, :, a].ravel(), extra[:, a]], 10))
                            for a in range(3)]
        shape = tuple(len(c)-1 for c in self.coordinates)
        if np.prod(shape) > 8_000_000:
            raise ValueError('Coordinate arrangement exceeds supported memory budget')
        self.eps = np.full(shape, np.nan)
        self.metal = np.full(shape, -1, np.int16)
        for r in records:
            eps = float(r['epsilon_r'])
            if eps <= 0 or not np.isfinite(eps):
                raise ValueError('Positive finite permittivity required')
            self.eps[self._slices(r['bounds'])] = eps
        if not np.isfinite(self.eps).all():
            raise ValueError('Material records do not cover the window')
        for b, tag in zip(self.boxes, self.tags):
            self.metal[self._slices(b)] = tag
        # The arbitrary extension through metal has no physical effect: both
        # the potential and each lift are constant in every metal component.
        self.eps[self.metal >= 0] = 1.
        self.interfaces = self._interfaces()

    def _slices(self, bounds, nodes=False):
        b = np.round(bounds, 10)
        return tuple(slice(np.searchsorted(c, b[0, a]), np.searchsorted(c, b[1, a])+(1 if nodes else 0))
                     for a, c in enumerate(self.coordinates))

    def _interfaces(self):
        unique = np.unique(self.eps)
        labels = np.searchsorted(unique, self.eps)
        patches = []
        for a in range(3):
            tang = [k for k in range(3) if k != a]
            for k in range(1, len(self.coordinates[a])-1):
                lm, lp = np.take(labels, k-1, a), np.take(labels, k, a)
                mm, mp = np.take(self.metal, k-1, a), np.take(self.metal, k, a)
                mask = np.where((lm != lp) & (mm < 0) & (mp < 0), 1+lm*len(unique)+lp, 0)
                for i, ie, j, je, _ in _rectangles(mask):
                    patches.append([a, self.coordinates[a][k], self.coordinates[tang[0]][i],
                                    self.coordinates[tang[0]][ie], self.coordinates[tang[1]][j],
                                    self.coordinates[tang[1]][je]])
        return np.asarray(patches, float).reshape(-1, 6)

    def material_volumes(self):
        volume = np.einsum('i,j,k->ijk', *[np.diff(c) for c in self.coordinates])
        return {float(e): float(volume[(self.eps == e) & (self.metal < 0)].sum())
                for e in np.unique(self.eps[self.metal < 0])}

    def materials(self, points):
        p = np.asarray(points)
        ids = tuple(np.clip(np.searchsorted(c, p[:, a], side='right')-1, 0, len(c)-2)
                    for a, c in enumerate(self.coordinates))
        return self.eps[ids]

    def lift(self, terminal):
        # Tensor interpolation of a geometry-only distance ratio. No reference
        # field, fitted capacitance, or training data enters this construction.
        if not hasattr(self, '_node_distances'):
            grid = np.stack(np.meshgrid(*self.coordinates, indexing='ij'), axis=-1)
            distance = np.full((self.nt,)+grid.shape[:-1], np.inf)
            for b, tag in zip(self.boxes, self.tags):
                d = np.maximum(np.maximum(b[0]-grid, grid-b[1]), 0).max(axis=-1)
                np.minimum(distance[tag], d, out=distance[tag])
            self._node_distances = distance
            self._closest_node_net = distance.argmin(axis=0)
            sorted_distance = np.partition(distance, 1, axis=0)
            self._closest_node_distance = sorted_distance[0].copy()
            self._second_node_distance = sorted_distance[1].copy()
        d_own = self._node_distances[terminal]
        d_other = np.where(self._closest_node_net == terminal, self._second_node_distance, self._closest_node_distance)
        w = np.maximum(0, 1-d_own/self.padding)*d_other/(d_own+d_other)
        if not np.isfinite(w).all():
            raise ValueError('Undefined terminal lift')
        return w

    def control_values(self, points):
        p=np.asarray(points);index=[];fractions=[]
        for a,c in enumerate(self.coordinates):
            k=np.clip(np.searchsorted(c,p[:,a],side='right')-1,0,len(c)-2)
            index.append(k);fractions.append((p[:,a]-c[k])/(c[k+1]-c[k]))
        values=np.zeros((len(p),self.nt-1))
        for b in range(8):
            ids=tuple(index[a]+((b>>a)&1) for a in range(3))
            weight=np.prod([fractions[a] if b&(1<<a) else 1-fractions[a] for a in range(3)],axis=0)
            values+=weight[:,None]*self.control[ids]
        return values

    def control_charge(self, measure):
        # Bilinear density times bilinear control: exact tensor mass matrix.
        matrix=np.kron(np.array([[1/3,1/6],[1/6,1/3]]),np.array([[1/3,1/6],[1/6,1/3]]))
        result=np.zeros(self.nt-1)
        for a in range(3):
            select=measure.rectangles[:,0]==a;r=measure.rectangles[select]
            tang=[k for k in range(3) if k!=a]
            k=np.searchsorted(self.coordinates[a],r[:,1])
            i=np.searchsorted(self.coordinates[tang[0]],r[:,2]);j=np.searchsorted(self.coordinates[tang[1]],r[:,4])
            corner=[]
            for bi,bj in [(0,0),(0,1),(1,0),(1,1)]:
                ids=[None]*3;ids[a]=k;ids[tang[0]]=i+bi;ids[tang[1]]=j+bj
                corner.append(self.control[tuple(ids)])
            coefficient=measure.density[select]@matrix
            result+=np.einsum('p,pk,kpj->j',measure.areas[select],coefficient,np.array(corner))
        return result

    def charge_measure(self, terminal):
        w = self.lift(terminal)
        return self.measure_from_lift(w)

    def measure_from_lift(self, w):
        """Return rectangles, four signed Bernstein coefficients, known tags.

        Includes the outer trace +epsilon*d_n w for Neumann windows. Interior
        coefficients are epsilon_minus*d_a w_minus - epsilon_plus*d_a w_plus.
        Corner order is (0,0), (0,1), (1,0), (1,1).
        """
        rectangles, densities, terminals = [], [], []
        for a in range(3):
            tang = [k for k in range(3) if k != a]
            # Moving the differentiated axis first leaves tangent axes ordered.
            v = np.moveaxis(w, a, 0)
            derivative = np.diff(v, axis=0)/np.diff(self.coordinates[a])[:, None, None]
            eps = np.moveaxis(self.eps, a, 0)
            corner = np.stack([derivative[:, :-1, :-1], derivative[:, :-1, 1:],
                               derivative[:, 1:, :-1], derivative[:, 1:, 1:]], -1)
            flux = eps[..., None]*corner
            density = np.concatenate([-flux[:1], flux[:-1]-flux[1:], flux[-1:]], axis=0)
            metal = np.moveaxis(self.metal, a, 0)
            known = np.concatenate([metal[:1], np.maximum(metal[:-1], metal[1:]), metal[-1:]], axis=0)
            ids = np.argwhere(np.any(abs(density) > 1e-12, axis=-1))
            k, i, j = ids.T
            rect = np.column_stack([np.full(len(ids), a), self.coordinates[a][k],
                                    self.coordinates[tang[0]][i], self.coordinates[tang[0]][i+1],
                                    self.coordinates[tang[1]][j], self.coordinates[tang[1]][j+1]])
            rectangles.append(rect); densities.append(density[k, i, j]); terminals.append(known[k, i, j])
        return FaceMeasure(np.concatenate(rectangles), np.concatenate(densities), np.concatenate(terminals), self.nt)


class FaceMeasure:
    def __init__(self, rectangles, density, known, nt):
        self.rectangles, self.density, self.known, self.nt = rectangles, density, known, nt
        self.areas = (rectangles[:, 3]-rectangles[:, 2])*(rectangles[:, 5]-rectangles[:, 4])
        self.mass = self.areas*density.sum(axis=1)/4
        self.exact = np.bincount(known[known >= 0], weights=self.mass[known >= 0], minlength=nt)
        # Each bilinear Bernstein basis is nonnegative. Sampling its corner
        # components avoids any sign approximation or rejection discretization.
        self.free = np.flatnonzero(known < 0)
        component = self.areas[self.free, None]*abs(density[self.free])/4
        self.total_variation_bound = float(component.sum())
        self.cdf = np.cumsum(component.ravel())
        if len(self.cdf) and self.cdf[-1] > 0:
            self.cdf /= self.cdf[-1]

    def sample(self, n, rng):
        if self.total_variation_bound == 0:
            raise ValueError('No unknown measure to sample')
        index = np.searchsorted(self.cdf, rng.random(n))
        face, corner = self.free[index//4], index % 4
        r = self.rectangles[face]
        a = r[:, 0].astype(int)
        t0 = np.where(a == 0, 1, 0); t1 = np.where(a == 2, 1, 2)
        z = np.sqrt(rng.random((n, 2)))
        z[:, 0] = np.where(corner//2, z[:, 0], 1-z[:, 0])
        z[:, 1] = np.where(corner % 2, z[:, 1], 1-z[:, 1])
        x = np.empty((n, 3)); rows = np.arange(n)
        x[rows, a] = r[:, 1]
        x[rows, t0] = r[:, 2]+z[:, 0]*(r[:, 3]-r[:, 2])
        x[rows, t1] = r[:, 4]+z[:, 1]*(r[:, 5]-r[:, 4])
        sign = np.sign(self.density[face, corner])
        return x, sign*self.total_variation_bound

    def integrate(self, function, order=3):
        g, q = np.polynomial.legendre.leggauss(order); g = (g+1)/2; q = q/2
        r = self.rectangles; a = r[:, 0].astype(int); rows = np.arange(len(r))
        t0 = np.where(a == 0, 1, 0); t1 = np.where(a == 2, 1, 2)
        total = None
        for i, u in enumerate(g):
            for j, v in enumerate(g):
                x = np.empty((len(r), 3)); x[rows, a] = r[:, 1]
                x[rows, t0] = r[:, 2]+u*(r[:, 3]-r[:, 2])
                x[rows, t1] = r[:, 4]+v*(r[:, 5]-r[:, 4])
                rho = self.density@np.array([(1-u)*(1-v), (1-u)*v, u*(1-v), u*v])
                values = function(x)
                term = np.tensordot(q[i]*q[j]*self.areas*rho, values, axes=1)
                total = term if total is None else total+term
        return total
