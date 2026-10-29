#include <wosx/wosx_gpu.h>
#include <wosx/utils/fcpw_geometric_queries.h>
#include <happly.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Vec4 = std::array<float, 4>;
using Vec3 = fcpw::Vector3;
using Tri = fcpw::Vector3i;
struct EmptyEntry : fcpw::GPUShaderEntryPoint {
    void setResources(const fcpw::ShaderCursor&, bool) const override {}
};

static void loadMesh(const std::string& filename, std::vector<Vec3>& vertices,
                     std::vector<Tri>& triangles) {
    const auto ext = std::filesystem::path(filename).extension().string();
    if (ext == ".obj") {
        wosx::loadBoundaryMesh<3>(filename, vertices, triangles);
    } else if (ext == ".ply") {
        happly::PLYData ply(filename);
        auto points = ply.getVertexPositions();
        auto faces = ply.getFaceIndices<size_t>();
        vertices.reserve(points.size());
        for (const auto& p : points) vertices.emplace_back(float(p[0]), float(p[1]), float(p[2]));
        for (const auto& f : faces) {
            if (f.size() != 3) throw std::runtime_error("PLY faces must be triangles");
            for (size_t i = 1; i + 1 < f.size(); ++i) {
                if (f[0] >= vertices.size() || f[i] >= vertices.size() || f[i+1] >= vertices.size())
                    throw std::runtime_error("Face index outside vertex array");
                triangles.emplace_back(int(f[0]), int(f[i]), int(f[i+1]));
            }
        }
    } else throw std::runtime_error("Triangle mesh must be .ply or .obj");
    if (vertices.empty() || triangles.empty()) throw std::runtime_error("Empty triangle mesh");
    for (const auto& v : vertices) for (int a=0; a<3; ++a)
        if (!std::isfinite(v[a])) throw std::runtime_error("Nonfinite mesh vertex");
}

template<class T> static std::vector<T> readArray(std::ifstream& stream, size_t count) {
    std::vector<T> out(count);
    stream.read(reinterpret_cast<char*>(out.data()), count*sizeof(T));
    if (!stream) throw std::runtime_error("Incomplete point fixture");
    return out;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4 && argc != 13) {
        std::cerr << "usage: triangle_mesh_walk_gpu MESH --info | REPO_ROOT MESH INPUT OUTPUT EPS_LOW EPS_HIGH INTERFACE_Z PADDING MAX_STEPS SEED TOLERANCE CONTINUATIONS\n";
        return 2;
    }
    bool info = argc == 3;
    bool dump = argc == 4 && std::string(argv[2]) == "--dump";
    if (argc == 4 && !dump) throw std::runtime_error("Expected --dump MESH_BIN");
    const std::string meshFile = (info || dump) ? argv[1] : argv[2];
    std::vector<Vec3> vertices;
    std::vector<Tri> triangles;
    loadMesh(meshFile, vertices, triangles);
    std::array<float,3> lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
    for (const auto& p : vertices) for (int a=0; a<3; ++a) {
        lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]);
    }
    if (dump) {
        std::ofstream out(argv[3], std::ios::binary);
        const uint32_t counts[2] = {uint32_t(vertices.size()), uint32_t(triangles.size())};
        out.write(reinterpret_cast<const char*>(counts), sizeof(counts));
        for (const auto& v : vertices) {
            const float xyz[3] = {v[0], v[1], v[2]};
            out.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
        }
        for (const auto& t : triangles) {
            const uint32_t ijk[3] = {uint32_t(t[0]), uint32_t(t[1]), uint32_t(t[2])};
            out.write(reinterpret_cast<const char*>(ijk), sizeof(ijk));
        }
        if (!out) throw std::runtime_error("Could not dump mesh");
        return 0;
    }
    if (info) {
        std::cout << "{\"vertices\":" << vertices.size() << ",\"triangles\":" << triangles.size()
                  << ",\"low\":[" << lo[0] << ',' << lo[1] << ',' << lo[2]
                  << "],\"high\":[" << hi[0] << ',' << hi[1] << ',' << hi[2] << "]}\n";
        return 0;
    }
    const std::filesystem::path root(argv[1]);
    const float epsLow = std::stof(argv[5]), epsHigh = std::stof(argv[6]);
    const float interfaceZ = std::stof(argv[7]), padding = std::stof(argv[8]);
    const uint32_t maxSteps = uint32_t(std::stoul(argv[9]));
    const uint32_t seed = uint32_t(std::stoul(argv[10]));
    const float tolerance = std::stof(argv[11]);
    const uint32_t continuations = uint32_t(std::stoul(argv[12]));
    if (!(epsLow>0 && epsHigh>0 && padding>0 && tolerance>0) || maxSteps==0)
        throw std::runtime_error("Invalid PDE or walk parameters");
    float span = std::max({hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]});
    if (!(span > 0)) throw std::runtime_error("Degenerate mesh bounds");
    std::array<float,3> outerLow, outerHigh;
    for (int a=0; a<3; ++a) { outerLow[a]=lo[a]-padding*span; outerHigh[a]=hi[a]+padding*span; }
    if (interfaceZ <= outerLow[2] || interfaceZ >= outerHigh[2])
        throw std::runtime_error("Dielectric interface lies outside the outer box");
    std::ifstream stream(argv[3], std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open point fixture");
    uint32_t count;
    stream.read(reinterpret_cast<char*>(&count),sizeof(count));
    if (!stream || count==0 || count>1000000) throw std::runtime_error("Invalid point count");
    auto points=readArray<Vec4>(stream,count);
    fcpw::GPUContext context;
    context.searchPaths={(root/"include/wosx/gpu").string(),(root/"deps/fcpw/include/fcpw/gpu").string()};
    context.macros={{"WOSX_PROBLEM_DIMENSION","3"},{"WOSX_DATA_CHANNELS","1"}};
    context.initDevice(fcpw::DeviceType::Default);
    wosx::GPUFcpwDirichletBoundaryHandler<3> mesh(vertices,triangles,false);
    mesh.allocate(context);
    fcpw::GPUModule module;
    module.load(context,(root/"demo_apps/heterogeneous_layers/triangle_mesh_walk_gpu.slang").string());
    fcpw::ComputeShader shader;
    shader.loadProgram(context,module,{},"runTriangleMeshWalk");
    fcpw::GPUBuffer pointsGpu,resultsGpu,endpointsGpu;
    pointsGpu.allocate(context,false,points);
    resultsGpu.allocate(context,true,std::vector<Vec4>(count));
    endpointsGpu.allocate(context,true,std::vector<Vec4>(count));
    const fcpw::float3 low{outerLow[0],outerLow[1],outerLow[2]};
    const fcpw::float3 high{outerHigh[0],outerHigh[1],outerHigh[2]};
    std::vector<Vec4> results;
    uint32_t completed=0;
    for (uint32_t pass=0; pass<=continuations; ++pass) {
        fcpw::runShader(context,shader,EmptyEntry{},[&](const fcpw::ComputeShader& program,const fcpw::ShaderCursor& cursor){
            auto meshObject=program.createShaderObject(context,mesh.getReflectionType());
            mesh.setResources(fcpw::ShaderCursor(meshObject),false);
            cursor.getPath("gMesh").setObject(meshObject);
            cursor.getPath("gStarts").setBinding(pointsGpu.buffer);
            cursor.getPath("gResults").setBinding(resultsGpu.buffer);
            cursor.getPath("gEndpoints").setBinding(endpointsGpu.buffer);
            cursor.getPath("gCount").setData(count);
            cursor.getPath("gMaxSteps").setData(maxSteps);
            cursor.getPath("gSeed").setData(seed);
            cursor.getPath("gPass").setData(pass);
            cursor.getPath("gTolerance").setData(tolerance);
            cursor.getPath("gLow").setData(low);
            cursor.getPath("gHigh").setData(high);
            cursor.getPath("gInterfaceZ").setData(interfaceZ);
            cursor.getPath("gEpsLow").setData(epsLow);
            cursor.getPath("gEpsHigh").setData(epsHigh);
        },(count+63)/64,1,false);
        resultsGpu.read(context,results);
        completed=0;
        for (const auto& r:results) completed += r[0]>=0;
        if (completed==count) break;
    }
    std::vector<Vec4> endpoints;
    endpointsGpu.read(context,endpoints);
    std::ofstream output(argv[4],std::ios::binary);
    output.write(reinterpret_cast<const char*>(&count),sizeof(count));
    output.write(reinterpret_cast<const char*>(results.data()),count*sizeof(Vec4));
    output.write(reinterpret_cast<const char*>(endpoints.data()),count*sizeof(Vec4));
    if (!output) throw std::runtime_error("Incomplete output");
    std::cout << "walks: " << completed << '/' << count << " reached a boundary\n";
    return 0;
}
