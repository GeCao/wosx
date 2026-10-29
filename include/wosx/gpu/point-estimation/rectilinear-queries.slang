/* SPDX-License-Identifier: Apache-2.0 */
implementing wosx;

// Packed box: low.xyz, low.w = terminal tag, high.xyz. Exact box distance
// avoids triangulating the case-8 metal unions.
public struct NearestRectilinearHit
{
    public float distance;
    public float3 projection;
    public int index;
    public int tag;
};

public NearestRectilinearHit nearestMetalBox(
    float3 point, StructuredBuffer<float4> lows,
    StructuredBuffer<float4> highs, uint count)
{
    NearestRectilinearHit hit;
    hit.distance = 3.402823466e+38F;
    hit.projection = point;
    hit.index = -1;
    hit.tag = -1;
    for (uint i = 0; i < count; ++i)
    {
        float3 projection = clamp(point, lows[i].xyz, highs[i].xyz);
        float distance = length(projection - point);
        if (distance < hit.distance)
        {
            hit.distance = distance;
            hit.projection = projection;
            hit.index = int(i);
            hit.tag = int(lows[i].w);
        }
    }
    return hit;
}

// Packed patch: a=(axis, normal coordinate, first low, first high),
// b=(second low, second high, 0, 0). The tangent-axis order is increasing.
public NearestRectilinearHit nearestInterfaceRectangle(
    float3 point, StructuredBuffer<float4> first,
    StructuredBuffer<float4> second, uint count)
{
    NearestRectilinearHit hit;
    hit.distance = 3.402823466e+38F;
    hit.projection = point;
    hit.index = -1;
    hit.tag = -1;
    for (uint i = 0; i < count; ++i)
    {
        float4 a = first[i];
        float4 b = second[i];
        int axis = int(a.x);
        float3 projection = point;
        projection[axis] = a.y;
        int firstTangent = axis == 0 ? 1 : 0;
        int secondTangent = axis == 2 ? 1 : 2;
        projection[firstTangent] = clamp(point[firstTangent], a.z, a.w);
        projection[secondTangent] = clamp(point[secondTangent], b.x, b.y);
        float distance = length(projection - point);
        if (distance < hit.distance)
        {
            hit.distance = distance;
            hit.projection = projection;
            hit.index = int(i);
        }
    }
    return hit;
}

// Coordinate arrangement is exact: each cell has a constant dielectric value.
// Flattening uses ((x * ny) + y) * nz + z, matching NumPy C order.
public uint coordinateCell(float value, StructuredBuffer<float> axis, uint cellCount)
{
    uint low = 0;
    uint high = cellCount;
    while (low < high)
    {
        uint middle = (low + high) / 2;
        if (value >= axis[middle + 1]) low = middle + 1;
        else high = middle;
    }
    return min(low, cellCount - 1);
}

public float rectilinearPermittivity(
    float3 point, StructuredBuffer<float> axisX,
    StructuredBuffer<float> axisY, StructuredBuffer<float> axisZ,
    StructuredBuffer<float> epsilon, uint nx, uint ny, uint nz)
{
    uint x = coordinateCell(point.x, axisX, nx);
    uint y = coordinateCell(point.y, axisY, ny);
    uint z = coordinateCell(point.z, axisZ, nz);
    return epsilon[(x * ny + y) * nz + z];
}

// Reuse FCPW's GPU BVH topology, but evaluate the original rectangle at each
// leaf. This avoids loss of accuracy from projecting onto very thin triangles.
public NearestRectilinearHit nearestInterfaceRectangleBvh(
    float3 point, Bvh<BvhNode, Triangle, NoSilhouette> bvh,
    StructuredBuffer<float4> first, StructuredBuffer<float4> second,
    bool skipIncident = false)
{
    NearestRectilinearHit hit;
    hit.distance = 3.402823466e+38F;
    hit.projection = point;
    hit.index = -1;
    hit.tag = -1;
    float bestSquared = hit.distance;
    uint stack[128];
    int top = 0;
    stack[0] = 0;
    while (top >= 0)
    {
        uint nodeIndex = stack[top--];
        BvhNode node = bvh.nodes[nodeIndex];
        float3 lower = max(max(node.box.pMin - point,
                               point - node.box.pMax), float3(0.0));
        if (dot(lower, lower) > bestSquared) continue;
        if (node.isLeaf())
        {
            for (uint local = 0; local < node.getNumPrimitives(); ++local)
            {
                uint triangleIndex = bvh.primitives[node.getPrimitiveOffset() + local].index;
                uint rectangleIndex = triangleIndex / 2;
                float4 a = first[rectangleIndex];
                float4 b = second[rectangleIndex];
                int axis = int(a.x);
                float3 q = point;
                q[axis] = a.y;
                int t0 = axis == 0 ? 1 : 0;
                int t1 = axis == 2 ? 1 : 2;
                q[t0] = clamp(point[t0], a.z, a.w);
                q[t1] = clamp(point[t1], b.x, b.y);
                float3 difference = q - point;
                float squared = dot(difference, difference);
                // Only interfaces through the center define radial sectors.
                // A disjoint coplanar patch must still bound the jump radius.
                if (skipIncident && squared < 1e-10) continue;
                if (squared < bestSquared)
                {
                    bestSquared = squared;
                    hit.projection = q;
                    hit.index = int(rectangleIndex);
                }
            }
        }
        else
        {
            uint left = nodeIndex + 1;
            uint right = nodeIndex + node.getRightChildOffset();
            float3 dl = max(max(bvh.nodes[left].box.pMin - point,
                                point - bvh.nodes[left].box.pMax), float3(0.0));
            float3 dr = max(max(bvh.nodes[right].box.pMin - point,
                                point - bvh.nodes[right].box.pMax), float3(0.0));
            float leftSquared = dot(dl, dl);
            float rightSquared = dot(dr, dr);
            if (leftSquared <= rightSquared)
            {
                if (rightSquared <= bestSquared) stack[++top] = right;
                if (leftSquared <= bestSquared) stack[++top] = left;
            }
            else
            {
                if (leftSquared <= bestSquared) stack[++top] = left;
                if (rightSquared <= bestSquared) stack[++top] = right;
            }
        }
    }
    hit.distance = sqrt(bestSquared);
    return hit;
}

// FCPW groups the twelve triangles of each box in its BVH. Outside metal,
// triangle bounds safely prune box-distance candidates. A containment pass
// handles interior points, where surface distance differs from box distance.
public NearestRectilinearHit nearestMetalBoxBvh(
    float3 point, Bvh<BvhNode, Triangle, NoSilhouette> bvh,
    StructuredBuffer<float4> lows, StructuredBuffer<float4> highs, uint count,
    float3 allLow, float3 allHigh)
{
    NearestRectilinearHit hit;
    hit.distance = 3.402823466e+38F;
    hit.projection = point;
    hit.index = -1;
    hit.tag = -1;
    if (all(point >= allLow) && all(point <= allHigh))
    for (uint i = 0; i < count; ++i)
    {
        if (all(point >= lows[i].xyz) && all(point <= highs[i].xyz))
        {
            hit.distance = 0.0;
            hit.index = int(i);
            hit.tag = int(lows[i].w);
            return hit;
        }
    }
    float bestSquared = hit.distance;
    uint stack[128];
    int top = 0;
    stack[0] = 0;
    while (top >= 0)
    {
        uint nodeIndex = stack[top--];
        BvhNode node = bvh.nodes[nodeIndex];
        float3 lower = max(max(node.box.pMin - point,
                               point - node.box.pMax), float3(0.0));
        if (dot(lower, lower) > bestSquared) continue;
        if (node.isLeaf())
        {
            for (uint local = 0; local < node.getNumPrimitives(); ++local)
            {
                uint triangleIndex = bvh.primitives[node.getPrimitiveOffset() + local].index;
                uint boxIndex = triangleIndex / 12;
                float3 q = clamp(point, lows[boxIndex].xyz, highs[boxIndex].xyz);
                float3 difference = q - point;
                float squared = dot(difference, difference);
                if (squared < bestSquared)
                {
                    bestSquared = squared;
                    hit.projection = q;
                    hit.index = int(boxIndex);
                    hit.tag = int(lows[boxIndex].w);
                }
            }
        }
        else
        {
            uint left = nodeIndex + 1;
            uint right = nodeIndex + node.getRightChildOffset();
            float3 dl = max(max(bvh.nodes[left].box.pMin - point,
                                point - bvh.nodes[left].box.pMax), float3(0.0));
            float3 dr = max(max(bvh.nodes[right].box.pMin - point,
                                point - bvh.nodes[right].box.pMax), float3(0.0));
            float leftSquared = dot(dl, dl);
            float rightSquared = dot(dr, dr);
            if (leftSquared <= rightSquared)
            {
                if (rightSquared <= bestSquared) stack[++top] = right;
                if (leftSquared <= bestSquared) stack[++top] = left;
            }
            else
            {
                if (leftSquared <= bestSquared) stack[++top] = left;
                if (rightSquared <= bestSquared) stack[++top] = right;
            }
        }
    }
    hit.distance = sqrt(bestSquared);
    return hit;
}

// Compatibility entry point for query fixtures without a precomputed union box.
public NearestRectilinearHit nearestMetalBoxBvh(
    float3 point, Bvh<BvhNode, Triangle, NoSilhouette> bvh,
    StructuredBuffer<float4> lows, StructuredBuffer<float4> highs, uint count)
{
    return nearestMetalBoxBvh(point, bvh, lows, highs, count,
                              float3(-3.402823466e+38F),
                              float3(3.402823466e+38F));
}

public uint incidentInterfaceAxes(float3 point, StructuredBuffer<float4> first,
                                  StructuredBuffer<float4> second, uint count)
{
    uint mask = 0;
    for (uint i = 0; i < count; ++i)
    {
        float4 a = first[i];
        float4 b = second[i];
        int axis = int(a.x);
        int t0 = axis == 0 ? 1 : 0;
        int t1 = axis == 2 ? 1 : 2;
        if (abs(point[axis] - a.y) <= 1e-5 &&
            point[t0] >= a.z - 1e-5 && point[t0] <= a.w + 1e-5 &&
            point[t1] >= b.x - 1e-5 && point[t1] <= b.y + 1e-5)
            mask |= 1u << uint(axis);
    }
    return mask;
}

public float snapCoordinate(float value, StructuredBuffer<float> axis,
                            uint cellCount, float tolerance)
{
    uint cell = coordinateCell(value, axis, cellCount);
    float lower = axis[cell];
    float upper = axis[cell + 1];
    float closest = abs(value - lower) <= abs(value - upper) ? lower : upper;
    return abs(value - closest) < tolerance ? closest : value;
}
