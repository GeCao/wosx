#include <wosx/wosx_gpu.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Vec4 = std::array<float, 4>;
using Vec3 = fcpw::Vector3;
using Tri = fcpw::Vector3i;

struct EmptyEntry : fcpw::GPUShaderEntryPoint {
    void setResources(const fcpw::ShaderCursor&, bool) const override {}
};

template <class T>
std::vector<T> readArray(std::ifstream& stream, size_t count) {
    std::vector<T> data(count);
    stream.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    if (!stream) throw std::runtime_error("Incomplete geometry fixture");
    return data;
}

static void addRectangle(int axis, float coordinate, float lo0, float hi0,
                         float lo1, float hi1, std::vector<Vec3>& vertices,
                         std::vector<Tri>& triangles) {
    const int tangent0 = axis == 0 ? 1 : 0;
    const int tangent1 = axis == 2 ? 1 : 2;
    const int first = int(vertices.size());
    for (int corner = 0; corner < 4; ++corner) {
        Vec3 vertex;
        vertex[axis] = coordinate;
        vertex[tangent0] = (corner == 1 || corner == 2) ? hi0 : lo0;
        vertex[tangent1] = corner >= 2 ? hi1 : lo1;
        vertices.push_back(vertex);
    }
    triangles.emplace_back(first, first + 1, first + 2);
    triangles.emplace_back(first, first + 2, first + 3);
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: rectilinear_walk_gpu REPO_ROOT FIXTURE_BIN OUTPUT_BIN MAX_STEPS SEED TOLERANCE INTERFACE_TOLERANCE CONTINUATIONS\n";
        return 2;
    }
    const std::filesystem::path root(argv[1]);
    std::ifstream stream(argv[2], std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open geometry fixture");
    const auto header = readArray<uint32_t>(stream, 8);
    const uint32_t nMetals = header[0], nInterfaces = header[1], nPoints = header[2];
    if (!nMetals || !nInterfaces || !nPoints || nMetals > 100000 ||
        nInterfaces > 100000 || nPoints > 1000000) {
        throw std::runtime_error("Invalid geometry fixture sizes");
    }
    const auto metalLow = readArray<Vec4>(stream, nMetals);
    const auto metalHigh = readArray<Vec4>(stream, nMetals);
    const auto patchFirst = readArray<Vec4>(stream, nInterfaces);
    const auto patchSecond = readArray<Vec4>(stream, nInterfaces);
    const auto points = readArray<Vec4>(stream, nPoints);
    const uint32_t nx = header[3], ny = header[4], nz = header[5];
    if (!nx || !ny || !nz || uint64_t(nx) * ny * nz > 8000000)
        throw std::runtime_error("Invalid permittivity grid shape");
    const auto axisX = readArray<float>(stream, nx + 1);
    const auto axisY = readArray<float>(stream, ny + 1);
    const auto axisZ = readArray<float>(stream, nz + 1);
    const auto epsilon = readArray<float>(stream, size_t(nx) * ny * nz);
    const auto expectedMetal = readArray<Vec4>(stream, nPoints);
    const auto expectedInterface = readArray<Vec4>(stream, nPoints);

    std::vector<Vec3> metalVertices, interfaceVertices;
    std::vector<Tri> metalTriangles, interfaceTriangles;
    for (size_t box = 0; box < nMetals; ++box) {
        for (int axis = 0; axis < 3; ++axis) {
            const int tangent0 = axis == 0 ? 1 : 0;
            const int tangent1 = axis == 2 ? 1 : 2;
            for (int side = 0; side < 2; ++side) {
                addRectangle(axis, side ? metalHigh[box][axis] : metalLow[box][axis],
                             metalLow[box][tangent0], metalHigh[box][tangent0],
                             metalLow[box][tangent1], metalHigh[box][tangent1],
                             metalVertices, metalTriangles);
            }
        }
    }
    for (size_t patch = 0; patch < nInterfaces; ++patch) {
        const auto& a = patchFirst[patch];
        const auto& b = patchSecond[patch];
        addRectangle(int(a[0]), a[1], a[2], a[3], b[0], b[1],
                     interfaceVertices, interfaceTriangles);
    }

    const uint32_t maxSteps = uint32_t(std::stoul(argv[4]));
    const uint32_t seed = uint32_t(std::stoul(argv[5]));
    const float tolerance = std::stof(argv[6]);
    const float interfaceTolerance = std::stof(argv[7]);
    const uint32_t continuations = uint32_t(std::stoul(argv[8]));
    if (!maxSteps || !(tolerance > 0) || !(interfaceTolerance > 0))
        throw std::runtime_error("Invalid walk settings");
    fcpw::GPUContext context;
    context.searchPaths = {(root / "include/wosx/gpu").string(),
                           (root / "deps/fcpw/include/fcpw/gpu").string()};
    context.macros = {{"WOSX_PROBLEM_DIMENSION", "3"}, {"WOSX_DATA_CHANNELS", "1"}};
    context.initDevice(fcpw::DeviceType::Default);
    wosx::GPUFcpwDirichletBoundaryHandler<3> metals(metalVertices, metalTriangles, false);
    wosx::GPUFcpwDirichletBoundaryHandler<3> interfaces(interfaceVertices, interfaceTriangles, false);
    metals.allocate(context);
    interfaces.allocate(context);
    fcpw::GPUModule module;
    module.load(context, (root / "demo_apps/heterogeneous_layers/rectilinear_walk_gpu.slang").string());
    fcpw::ComputeShader shader;
    shader.loadProgram(context, module, {}, "runRectilinearWalk");
    fcpw::GPUBuffer pointsGpu, metalLowGpu, metalHighGpu, patchFirstGpu, patchSecondGpu;
    fcpw::GPUBuffer axisXGpu, axisYGpu, axisZGpu, epsilonGpu, resultsGpu, endpointsGpu;
    pointsGpu.allocate(context, false, points);
    metalLowGpu.allocate(context, false, metalLow);
    metalHighGpu.allocate(context, false, metalHigh);
    patchFirstGpu.allocate(context, false, patchFirst);
    patchSecondGpu.allocate(context, false, patchSecond);
    axisXGpu.allocate(context, false, axisX);
    axisYGpu.allocate(context, false, axisY);
    axisZGpu.allocate(context, false, axisZ);
    epsilonGpu.allocate(context, false, epsilon);
    resultsGpu.allocate(context, true, std::vector<Vec4>(nPoints));
    endpointsGpu.allocate(context, true, std::vector<Vec4>(nPoints));
    std::array<float, 3> boundsLow{INFINITY, INFINITY, INFINITY};
    std::array<float, 3> boundsHigh{-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < metalLow.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            boundsLow[axis] = std::min(boundsLow[axis], metalLow[i][axis]);
            boundsHigh[axis] = std::max(boundsHigh[axis], metalHigh[i][axis]);
        }
    }
    const fcpw::float3 metalAllLow{boundsLow[0], boundsLow[1], boundsLow[2]};
    const fcpw::float3 metalAllHigh{boundsHigh[0], boundsHigh[1], boundsHigh[2]};
    const fcpw::float3 windowLow{axisX.front(), axisY.front(), axisZ.front()};
    const fcpw::float3 windowHigh{axisX.back(), axisY.back(), axisZ.back()};
    std::vector<Vec4> results;
    size_t completed = 0;
    for (uint32_t pass = 0; pass <= continuations; ++pass) {
    fcpw::runShader(context, shader, EmptyEntry{},
        [&](const fcpw::ComputeShader& program, const fcpw::ShaderCursor& cursor) {
            auto metalObject = program.createShaderObject(context, metals.getReflectionType());
            auto interfaceObject = program.createShaderObject(context, interfaces.getReflectionType());
            metals.setResources(fcpw::ShaderCursor(metalObject), false);
            interfaces.setResources(fcpw::ShaderCursor(interfaceObject), false);
            cursor.getPath("gMetalBvh").setObject(metalObject);
            cursor.getPath("gInterfaceBvh").setObject(interfaceObject);
            cursor.getPath("gMetalLow").setBinding(metalLowGpu.buffer);
            cursor.getPath("gMetalHigh").setBinding(metalHighGpu.buffer);
            cursor.getPath("gInterfaceFirst").setBinding(patchFirstGpu.buffer);
            cursor.getPath("gInterfaceSecond").setBinding(patchSecondGpu.buffer);
            cursor.getPath("gAxisX").setBinding(axisXGpu.buffer);
            cursor.getPath("gAxisY").setBinding(axisYGpu.buffer);
            cursor.getPath("gAxisZ").setBinding(axisZGpu.buffer);
            cursor.getPath("gEpsilon").setBinding(epsilonGpu.buffer);
            cursor.getPath("gStartPoints").setBinding(pointsGpu.buffer);
            cursor.getPath("gWalkResults").setBinding(resultsGpu.buffer);
            cursor.getPath("gEndpoints").setBinding(endpointsGpu.buffer);
            cursor.getPath("gMetalCount").setData(nMetals);
            cursor.getPath("gInterfaceCount").setData(nInterfaces);
            cursor.getPath("gPointCount").setData(nPoints);
            cursor.getPath("gNx").setData(nx);
            cursor.getPath("gNy").setData(ny);
            cursor.getPath("gNz").setData(nz);
            cursor.getPath("gMaxSteps").setData(maxSteps);
            cursor.getPath("gSeed").setData(seed);
            cursor.getPath("gPassIndex").setData(pass);
            cursor.getPath("gTolerance").setData(tolerance);
            cursor.getPath("gInterfaceTolerance").setData(interfaceTolerance);
            cursor.getPath("gWindowLow").setData(windowLow);
            cursor.getPath("gWindowHigh").setData(windowHigh);
            cursor.getPath("gMetalAllLow").setData(metalAllLow);
            cursor.getPath("gMetalAllHigh").setData(metalAllHigh);
        }, (nPoints + 63) / 64, 1, false);
    resultsGpu.read(context, results);
    completed = 0;
    for (const auto& row : results) completed += row[0] >= 0;
    if (completed == nPoints) break;
    }
    std::vector<Vec4> endpoints;
    endpointsGpu.read(context, endpoints);
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write walk output");
    output.write(reinterpret_cast<const char*>(&nPoints), sizeof(nPoints));
    output.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(Vec4));
    output.write(reinterpret_cast<const char*>(endpoints.data()), endpoints.size() * sizeof(Vec4));
    if (!output) throw std::runtime_error("Incomplete walk output");
    std::cout << "walks: " << completed << "/" << nPoints << " reached metal; output " << argv[3] << "\n";
    return 0;
}
