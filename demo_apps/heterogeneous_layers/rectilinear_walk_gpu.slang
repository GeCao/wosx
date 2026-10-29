import fcpw;
import wosx;

uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gMetalBvh;
uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gInterfaceBvh;
StructuredBuffer<float4> gMetalLow;
StructuredBuffer<float4> gMetalHigh;
StructuredBuffer<float4> gInterfaceFirst;
StructuredBuffer<float4> gInterfaceSecond;
StructuredBuffer<float> gAxisX;
StructuredBuffer<float> gAxisY;
StructuredBuffer<float> gAxisZ;
StructuredBuffer<float> gEpsilon;
StructuredBuffer<float4> gStartPoints;
RWStructuredBuffer<float4> gWalkResults;
RWStructuredBuffer<float4> gEndpoints;
uniform uint gMetalCount;
uniform uint gInterfaceCount;
uniform uint gPointCount;
uniform uint gNx;
uniform uint gNy;
uniform uint gNz;
uniform uint gMaxSteps;
uniform uint gSeed;
uniform uint gPassIndex;
uniform float gTolerance;
uniform float gInterfaceTolerance;
uniform float3 gWindowLow;
uniform float3 gWindowHigh;
uniform float3 gMetalAllLow;
uniform float3 gMetalAllHigh;

struct GridMaterial : IHeterogeneousMaterial3D
{
    float relativePermittivity(float3 point)
    {
        return rectilinearPermittivity(point, gAxisX, gAxisY, gAxisZ,
                                       gEpsilon, gNx, gNy, gNz);
    }
};

float3 snapInterfacePoint(float3 point)
{
    point.x = snapCoordinate(point.x, gAxisX, gNx, gInterfaceTolerance);
    point.y = snapCoordinate(point.y, gAxisY, gNy, gInterfaceTolerance);
    point.z = snapCoordinate(point.z, gAxisZ, gNz, gInterfaceTolerance);
    return point;
}

float jumpRadius(float3 point)
{
    NearestRectilinearHit metal = nearestMetalBoxBvh(
        point, gMetalBvh, gMetalLow, gMetalHigh, gMetalCount,
        gMetalAllLow, gMetalAllHigh);
    NearestRectilinearHit interfaceHit = nearestInterfaceRectangleBvh(
        point, gInterfaceBvh, gInterfaceFirst, gInterfaceSecond, true);
    float radius = min(metal.distance, interfaceHit.distance);
    // A window limit can only shorten the sphere. Applying it on each axis
    // avoids a full scan of every interface patch at every dielectric exit.
    for (int axis = 0; axis < 3; ++axis)
    {
        if (point[axis] > gWindowLow[axis] + 1e-5)
            radius = min(radius, 2.0 * (point[axis] - gWindowLow[axis]));
        if (point[axis] < gWindowHigh[axis] - 1e-5)
            radius = min(radius, 2.0 * (gWindowHigh[axis] - point[axis]));
    }
    return radius;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void runRectilinearWalk(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gPointCount) return;
    pcg32 rng;
    if (gPassIndex > 0 && gWalkResults[tid.x].x >= 0.0) return;
    uint pathId = uint(gStartPoints[tid.x].w);
    rng.seed(uint64_t(gSeed) + uint64_t(pathId) * 0x9e3779b97f4a7c15ULL
             + uint64_t(gPassIndex) * 0xbf58476d1ce4e5b9ULL);
    float3 point = gPassIndex == 0 ? gStartPoints[tid.x].xyz : gEndpoints[tid.x].xyz;
    int label = -2;
    uint steps = 0;
    uint crossings = 0;
    uint reflections = 0;
    if (gPassIndex > 0)
    {
        float4 previous = gWalkResults[tid.x];
        steps = uint(previous.y);
        crossings = uint(previous.z);
        reflections = uint(previous.w);
    }
    uint thisPassSteps = gMaxSteps;
    for (uint step = 0; step < gMaxSteps; ++step)
    {
        NearestRectilinearHit metal = nearestMetalBoxBvh(
            point, gMetalBvh, gMetalLow, gMetalHigh, gMetalCount,
        gMetalAllLow, gMetalAllHigh);
        if (metal.distance <= gTolerance)
        {
            label = metal.tag;
            thisPassSteps = step;
            break;
        }
        NearestRectilinearHit interfaceHit = nearestInterfaceRectangleBvh(
            point, gInterfaceBvh, gInterfaceFirst, gInterfaceSecond);
        float radius = min(metal.distance, interfaceHit.distance);
        float coordinateScale = max(1.0, max(abs(point.x), max(abs(point.y), abs(point.z))));
        float numericalFloor = min(gInterfaceTolerance, 2e-6 * coordinateScale);
        float nearInterface = min(gInterfaceTolerance,
                                  max(0.005 * metal.distance, numericalFloor));
        bool jump = interfaceHit.distance <= nearInterface;
        float3 center = point;
        if (jump)
        {
            float3 raw = interfaceHit.projection;
            center = snapInterfacePoint(raw);
            radius = jumpRadius(center);
            if (distance(center, point) > max(0.01 * radius, numericalFloor))
            {
                center = raw;
                radius = jumpRadius(center);
            }
            if (distance(center, point) > max(0.01 * radius, numericalFloor))
            {
                jump = false;
                center = point;
                radius = min(metal.distance, interfaceHit.distance);
            }
        }
        if (!(radius > 0.0) || !isfinite(radius))
        {
            thisPassSteps = step;
            break;
        }
        radius *= 0.99999994;
        float3 direction = SphereSampler3D.sampleUnitSphereUniform(rng);
        float3 next = jump ? sampleHeterogeneousInterfaceExit(
            center, radius, direction, gWindowLow, gWindowHigh,
            GridMaterial(), rng) : point + radius * direction;
        if (jump) ++crossings;
        if (any(next < gWindowLow) || any(next > gWindowHigh)) ++reflections;
        point = foldInsulatingWindow(next, gWindowLow, gWindowHigh);
    }
    steps += thisPassSteps;
    gWalkResults[tid.x] = float4(float(label), float(steps),
                                  float(crossings), float(reflections));
    gEndpoints[tid.x] = float4(point, 0.0);
}
