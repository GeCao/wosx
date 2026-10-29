/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module wosx;
__include utils.float_array;
__include utils.dense_grid;
__include utils.fcpw_geometric_queries;
__include utils.sdf_grid_geometric_queries;
__include point_estimation.walk_on_spheres;
__include point_estimation.heterogeneous_interface;
__include point_estimation.rectilinear_queries;
__include point_estimation.walk_on_stars;
__include variance_reduction.boundary_samplers;
__include variance_reduction.domain_samplers;
__include variance_reduction.boundary_value_caching;
__include variance_reduction.kelvin_transform;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem dimension defines

#define WOSX_UNDEFINED_PROBLEM_DIMENSION 0

#ifndef WOSX_PROBLEM_DIMENSION
#define WOSX_PROBLEM_DIMENSION WOSX_UNDEFINED_PROBLEM_DIMENSION
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem data type defines and typedefs

#define WOSX_UNDEFINED_DATA_CHANNELS 0

#ifndef WOSX_DATA_CHANNELS
#define WOSX_DATA_CHANNELS WOSX_UNDEFINED_DATA_CHANNELS
#endif

#if WOSX_DATA_CHANNELS == 1
public typedef float DataType;

#elif WOSX_DATA_CHANNELS > 4
public typedef FloatArray<WOSX_DATA_CHANNELS> DataType;

#else
public typedef vector<float, WOSX_DATA_CHANNELS> DataType;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Absorbing boundary geometric queries defines and typedefs

#define WOSX_EMPTY_ABSORBING_BOUNDARY 0
#define WOSX_MESH_BASED_ABSORBING_BOUNDARY 1
#define WOSX_SDF_BASED_ABSORBING_BOUNDARY 2

#ifndef WOSX_ABSORBING_BOUNDARY_TYPE
#define WOSX_ABSORBING_BOUNDARY_TYPE WOSX_EMPTY_ABSORBING_BOUNDARY
#endif

#if WOSX_ABSORBING_BOUNDARY_TYPE == WOSX_MESH_BASED_ABSORBING_BOUNDARY
#if WOSX_PROBLEM_DIMENSION == 2
typedef Bvh<BvhNode, LineSegment, NoSilhouette> AbsorbingBoundaryAccelerationStructure;

#elif WOSX_PROBLEM_DIMENSION == 3
typedef Bvh<BvhNode, Triangle, NoSilhouette> AbsorbingBoundaryAccelerationStructure;

#endif

typedef FcpwAbsorbingBoundaryGeometricQueries<AbsorbingBoundaryAccelerationStructure, WOSX_PROBLEM_DIMENSION> AbsorbingBoundaryGeometricQueries;

#elif WOSX_ABSORBING_BOUNDARY_TYPE == WOSX_SDF_BASED_ABSORBING_BOUNDARY
#if WOSX_PROBLEM_DIMENSION == 2
typedef SdfGrid2D AbsorbingBoundaryAccelerationStructure;

#elif WOSX_PROBLEM_DIMENSION == 3
typedef SdfGrid3D AbsorbingBoundaryAccelerationStructure;

#endif

typedef SdfAbsorbingBoundaryGeometricQueries<AbsorbingBoundaryAccelerationStructure, WOSX_PROBLEM_DIMENSION> AbsorbingBoundaryGeometricQueries;

#else
typedef EmptyAbsorbingBoundaryGeometricQueries<WOSX_PROBLEM_DIMENSION> AbsorbingBoundaryGeometricQueries;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Reflecting boundary geometric queries defines and typedefs

#define WOSX_EMPTY_REFLECTING_BOUNDARY 0
#define WOSX_MESH_BASED_REFLECTING_NEUMANN_BOUNDARY 1
#define WOSX_MESH_BASED_REFLECTING_ROBIN_BOUNDARY 2

#ifndef WOSX_REFLECTING_BOUNDARY_TYPE
#define WOSX_REFLECTING_BOUNDARY_TYPE WOSX_EMPTY_REFLECTING_BOUNDARY
#endif

#if WOSX_REFLECTING_BOUNDARY_TYPE == WOSX_MESH_BASED_REFLECTING_NEUMANN_BOUNDARY
#if WOSX_PROBLEM_DIMENSION == 2
typedef Bvh<SnchNode, LineSegment, Vertex> ReflectingBoundaryAccelerationStructure;

#elif WOSX_PROBLEM_DIMENSION == 3
typedef Bvh<SnchNode, Triangle, Edge> ReflectingBoundaryAccelerationStructure;

#endif

typedef FcpwReflectingBoundaryGeometricQueries<ReflectingBoundaryAccelerationStructure, WOSX_PROBLEM_DIMENSION> ReflectingBoundaryGeometricQueries;

#elif WOSX_REFLECTING_BOUNDARY_TYPE == WOSX_MESH_BASED_REFLECTING_ROBIN_BOUNDARY
#if WOSX_PROBLEM_DIMENSION == 2
typedef ReflectanceBvh<ReflectanceLineSegment<RobinLineSegmentBound>, RobinBvhNodeBound2D> ReflectingBoundaryAccelerationStructure;

#elif WOSX_PROBLEM_DIMENSION == 3
typedef ReflectanceBvh<ReflectanceTriangle<RobinTriangleBound>, RobinBvhNodeBound3D> ReflectingBoundaryAccelerationStructure;

#endif

typedef FcpwReflectingBoundaryGeometricQueries<ReflectingBoundaryAccelerationStructure, WOSX_PROBLEM_DIMENSION> ReflectingBoundaryGeometricQueries;

#else
typedef EmptyReflectingBoundaryGeometricQueries<WOSX_PROBLEM_DIMENSION> ReflectingBoundaryGeometricQueries;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Geometric queries typedef

public typedef GeometricQueries<AbsorbingBoundaryGeometricQueries, ReflectingBoundaryGeometricQueries, WOSX_PROBLEM_DIMENSION> GeometricQueriesType;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Solve region defines and typedefs

#define WOSX_UNDEFINED_SOLVE_REGION 0
#define WOSX_BOUNDING_BOX_SOLVE_REGION 1
#define WOSX_WATERTIGHT_DOMAIN_SOLVE_REGION 2

#ifndef WOSX_SOLVE_REGION_TYPE
#define WOSX_SOLVE_REGION_TYPE WOSX_UNDEFINED_SOLVE_REGION
#endif

#if WOSX_SOLVE_REGION_TYPE == WOSX_BOUNDING_BOX_SOLVE_REGION
typedef BoundingBoxSolveRegion<WOSX_PROBLEM_DIMENSION> SolveRegionType;

#elif WOSX_SOLVE_REGION_TYPE == WOSX_WATERTIGHT_DOMAIN_SOLVE_REGION
typedef WatertightDomainSolveRegion<GeometricQueriesType, WOSX_PROBLEM_DIMENSION> SolveRegionType;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Domain samplers defines and typedefs

#define WOSX_EMPTY_DOMAIN_SAMPLER 0
#define WOSX_UNIFORM_DOMAIN_SAMPLER 1

#ifndef WOSX_DOMAIN_SAMPLER_TYPE
#define WOSX_DOMAIN_SAMPLER_TYPE WOSX_EMPTY_DOMAIN_SAMPLER
#endif

#if WOSX_DOMAIN_SAMPLER_TYPE == WOSX_UNIFORM_DOMAIN_SAMPLER
public typedef UniformDomainSampler<SolveRegionType, GeometricQueriesType, WOSX_PROBLEM_DIMENSION> DomainSamplerType;

#else
public typedef EmptyDomainSampler<WOSX_PROBLEM_DIMENSION> DomainSamplerType;

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Boundary samplers defines and typedefs

#define WOSX_EMPTY_BOUNDARY_SAMPLER 0
#define WOSX_UNIFORM_LINE_SEGMENT_BOUNDARY_SAMPLER 1
#define WOSX_UNIFORM_TRIANGLE_BOUNDARY_SAMPLER 2

#ifndef WOSX_ABSORBING_BOUNDARY_SAMPLER_TYPE
#define WOSX_ABSORBING_BOUNDARY_SAMPLER_TYPE WOSX_EMPTY_BOUNDARY_SAMPLER
#endif

#if WOSX_ABSORBING_BOUNDARY_SAMPLER_TYPE == WOSX_UNIFORM_LINE_SEGMENT_BOUNDARY_SAMPLER
public typedef UniformLineSegmentBoundarySampler<GeometricQueriesType> AbsorbingBoundarySamplerType;

#elif WOSX_ABSORBING_BOUNDARY_SAMPLER_TYPE == WOSX_UNIFORM_TRIANGLE_BOUNDARY_SAMPLER
public typedef UniformTriangleBoundarySampler<GeometricQueriesType> AbsorbingBoundarySamplerType;

#else
public typedef EmptyBoundarySampler<WOSX_PROBLEM_DIMENSION> AbsorbingBoundarySamplerType;

#endif

#ifndef WOSX_REFLECTING_BOUNDARY_SAMPLER_TYPE
#define WOSX_REFLECTING_BOUNDARY_SAMPLER_TYPE WOSX_EMPTY_BOUNDARY_SAMPLER
#endif

#if WOSX_REFLECTING_BOUNDARY_SAMPLER_TYPE == WOSX_UNIFORM_LINE_SEGMENT_BOUNDARY_SAMPLER
public typedef UniformLineSegmentBoundarySampler<GeometricQueriesType> ReflectingBoundarySamplerType;

#elif WOSX_REFLECTING_BOUNDARY_SAMPLER_TYPE == WOSX_UNIFORM_TRIANGLE_BOUNDARY_SAMPLER
public typedef UniformTriangleBoundarySampler<GeometricQueriesType> ReflectingBoundarySamplerType;

#else
public typedef EmptyBoundarySampler<WOSX_PROBLEM_DIMENSION> ReflectingBoundarySamplerType;

#endif
