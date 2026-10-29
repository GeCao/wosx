import fcpw;
import wosx;

uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gMesh;
StructuredBuffer<float4> gStarts;
RWStructuredBuffer<float4> gResults;
RWStructuredBuffer<float4> gEndpoints;
uniform uint gCount;
uniform uint gMaxSteps;
uniform uint gSeed;
uniform uint gPass;
uniform float gTolerance;
uniform float3 gLow;
uniform float3 gHigh;
uniform float gInterfaceZ;
uniform float gEpsLow;
uniform float gEpsHigh;

[shader("compute")]
[numthreads(64, 1, 1)]
void runTriangleMeshWalk(uint3 tid : SV_DispatchThreadID)
{
    uint id = tid.x;
    if (id >= gCount) return;
    if (gPass > 0 && gResults[id].x >= 0.0) return;
    pcg32 rng;
    uint pathId = uint(gStarts[id].w);
    rng.seed(uint64_t(gSeed) + uint64_t(pathId) * 0x9e3779b97f4a7c15ULL
             + uint64_t(gPass) * 0xbf58476d1ce4e5b9ULL);
    float3 point = gPass == 0 ? gStarts[id].xyz : gEndpoints[id].xyz;
    uint steps = gPass == 0 ? 0 : uint(gResults[id].y);
    uint crossings = gPass == 0 ? 0 : uint(gResults[id].z);
    int label = -1;
    for (uint step = 0; step < gMaxSteps; ++step)
    {
        BoundingSphere sphere;
        sphere.c = point;
        sphere.r2 = 3.402823466e+38F;
        Interaction interaction;
        bool found = gMesh.findClosestPoint(sphere, interaction, false);
        if (!found) break;
        float meshDistance = sqrt(max(0.0, sphere.r2));
        float3 wall = min(point - gLow, gHigh - point);
        float wallDistance = min(wall.x, min(wall.y, wall.z));
        if (meshDistance <= gTolerance) { label = 1; break; }
        if (wallDistance <= gTolerance) { label = 0; break; }
        float interfaceDistance = abs(point.z - gInterfaceZ);
        float radius = min(meshDistance, min(wallDistance, interfaceDistance));
        float nearInterface = min(gTolerance, max(0.005 * meshDistance, 2e-6));
        bool jump = interfaceDistance <= nearInterface;
        float3 center = point;
        if (jump)
        {
            center.z = gInterfaceZ;
            radius = min(meshDistance - interfaceDistance,
                         min(wallDistance - interfaceDistance, 3.402823466e+38F));
            // Re-query at the projected center: this also handles a mesh crossing the interface.
            sphere.c = center;
            sphere.r2 = 3.402823466e+38F;
            if (!gMesh.findClosestPoint(sphere, interaction, false)) break;
            float3 centerWall = min(center - gLow, gHigh - center);
            radius = min(sqrt(max(0.0, sphere.r2)),
                         min(centerWall.x, min(centerWall.y, centerWall.z)));
        }
        if (!(radius > 0.0) || !isfinite(radius)) break;
        float3 direction = SphereSampler3D.sampleUnitSphereUniform(rng);
        if (jump)
        {
            float pHigh = gEpsHigh / (gEpsLow + gEpsHigh);
            direction.z = rng.nextFloat() < pHigh ? abs(direction.z) : -abs(direction.z);
            ++crossings;
        }
        point = center + (radius * 0.99999994) * direction;
        ++steps;
    }
    gResults[id] = float4(float(label), float(steps), float(crossings), 0.0);
    gEndpoints[id] = float4(point, 0.0);
}
