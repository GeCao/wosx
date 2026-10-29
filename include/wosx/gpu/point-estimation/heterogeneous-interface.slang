/*
 * SPDX-License-Identifier: Apache-2.0
 * Permittivity-weighted sphere exits at rectilinear dielectric junctions.
 */
implementing wosx;
__include core.sampling;

// The caller provides the material lookup for the current box arrangement.
public interface IHeterogeneousMaterial3D
{
    float relativePermittivity(float3 point);
};

// The insulating rectangular window is represented by an even extension.
public float3 foldInsulatingWindow(float3 point, float3 low, float3 high)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        float width = high[axis] - low[axis];
        float period = 2.0 * width;
        float shifted = point[axis] - low[axis];
        float phase = shifted - period * floor(shifted / period);
        point[axis] = low[axis] + (phase <= width ? phase : period - phase);
    }
    return point;
}

// All eight octants have equal solid angle. The probability of an octant is
// therefore proportional to its dielectric constant (Chapter 5 sector law).
public float3 sampleHeterogeneousInterfaceExit<MaterialLookup>(
    float3 center, float radius, float3 unitDirection,
    float3 windowLow, float3 windowHigh,
    MaterialLookup materials, inout pcg32 rng)
    where MaterialLookup : IHeterogeneousMaterial3D
{
    float weights[8];
    float total = 0.0;
    float offset = radius * 0.5773502691896258;
    for (int sector = 0; sector < 8; ++sector)
    {
        float3 representative = center + float3(
            (sector & 1) != 0 ? offset : -offset,
            (sector & 2) != 0 ? offset : -offset,
            (sector & 4) != 0 ? offset : -offset);
        representative = foldInsulatingWindow(representative, windowLow, windowHigh);
        weights[sector] = max(0.0, materials.relativePermittivity(representative));
        total += weights[sector];
    }
    float selector = rng.nextFloat() * total;
    int chosen = 7;
    for (int sector = 0; sector < 7; ++sector)
    {
        if (selector < weights[sector])
        {
            chosen = sector;
            break;
        }
        selector -= weights[sector];
    }
    float3 exitPoint = center + radius * float3(
        (chosen & 1) != 0 ? abs(unitDirection.x) : -abs(unitDirection.x),
        (chosen & 2) != 0 ? abs(unitDirection.y) : -abs(unitDirection.y),
        (chosen & 4) != 0 ? abs(unitDirection.z) : -abs(unitDirection.z));
    return foldInsulatingWindow(exitPoint, windowLow, windowHigh);
}
