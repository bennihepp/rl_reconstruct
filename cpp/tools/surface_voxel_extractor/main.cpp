#include <iostream>
#include <unordered_set>
#include <boost/program_options.hpp>
#include <boost/optional.hpp>
//#include <pcl/io/ply_io.h>
#include <bh/mLib/mLibCoreMesh.h>
#include <bh/math/geometry.h>
#include <bh/mLib/mLibUtils.h>
#include <bh/math/continuous_grid3d.h>

using std::size_t;
using std::cout;
using std::endl;
using std::string;

//using pcl::PLYReader;
//using pcl::PolygonMesh;

using FloatType = float;
USE_FIXED_EIGEN_TYPES(FloatType);

void run(const string& mesh_filename, const FloatType voxel_size,
         const boost::optional<string>& output_filename,
         const boost::optional<string>& mesh_output_filename) {
//  // Reading with PCL (quite picky on file format)
//  PolygonMesh mesh;
//  Eigen::Vector4f origin;
//  Eigen::Quaternionf orientation;
//  int ply_version;
//  PLYReader reader;
//  int result = reader.read(filename, mesh, origin, orientation, ply_version);
//  if (result <= 0) {
//    throw std::runtime_error("Error reading mesh: " + filename);
//  }
//  cout << "Mesh has " << mesh.polygons.size() << " faces and "
//       << mesh.cloud.width * mesh.cloud.height << " vertices" << endl;
  // Reading with mLib
  ml::MeshDataf mesh;
  ml::MeshIOf::loadFromPLY(mesh_filename, mesh);
  cout << "Mesh has " << mesh.m_FaceIndicesVertices.size() << " faces and "
       << mesh.m_Vertices.size() << " vertices" << endl;

  const bh::ContinuousGrid3DUtilsf grid_utils(voxel_size, 0.5f);
  std::unordered_set<Vector3, bh::EigenHash<Vector3>> intersecting_voxels;

  // Iterate over all triangles
  for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); ++i) {
//    cout << "Processing triangle " << i + 1 << " out of " << mesh.m_FaceIndicesVertices.size() << endl;
    const ml::MeshDataf::Indices::Face& face = mesh.m_FaceIndicesVertices[i];
    BH_ASSERT_STR(face.size() == 3, "Can only handle triangle meshes");

    const Vector3 v1 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[0]]);
    const Vector3 v2 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[1]]);
    const Vector3 v3 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[2]]);
    const bh::Trianglef tri(v1, v2, v3);

    // Loop over all voxels in triangle bounding box
    const bh::BoundingBox3Df bbox = tri.boundingBox();
    const Vector3 min_voxel = grid_utils.getMinIntersectingVoxel(bbox);
    const Vector3 max_voxel = grid_utils.getMaxIntersectingVoxel(bbox);
    for (FloatType x = min_voxel(0); x <= max_voxel(0); x += grid_utils.getIncrement()) {
      for (FloatType y = min_voxel(1); y <= max_voxel(1); y += grid_utils.getIncrement()) {
        for (FloatType z = min_voxel(2); z <= max_voxel(2); z += grid_utils.getIncrement()) {
          const Vector3 voxel(x, y, z);
          if (intersecting_voxels.find(voxel) != intersecting_voxels.end()) {
            continue;
          }
          const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
          if (tri.intersects(voxel_bbox)) {
            intersecting_voxels.insert(voxel);
          }
        }
      }
    }
  }
  cout << "Found " << intersecting_voxels.size() << " intersecting voxels" << endl;

  if (output_filename) {
    std::ofstream out(*output_filename);
    if (!out) {
      throw std::runtime_error("ERROR: Unable to open output file: " + *output_filename);
    }
    for (const Vector3 &voxel : intersecting_voxels) {
      out << voxel(0) << " " << voxel(1) << " " << voxel(2) << endl;
    }
  }

  if (mesh_output_filename) {
    std::vector<ml::TriMeshf> output_meshes;
    for (const Vector3 &voxel : intersecting_voxels) {
      const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
      const ml::BoundingBox3f ml_bbox(bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMinimum()),
                                      bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMaximum()));
      const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox);
      output_meshes.push_back(output_mesh);
    }
    const ml::TriMeshf output_mesh = ml::Shapesf::unifyMeshes(output_meshes);
    ml::MeshIOf::saveToPLY(*mesh_output_filename, output_mesh.getMeshData());
  }
}

int main(int argc, const char** argv) {
  try {
    /** Define and parse the program options
     */
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
            ("help", "Print help messages")
            ("voxel-size", po::value<FloatType>()->default_value(0.2), "Voxel size")
            ("mesh-filename", po::value<string>()->required(), "Mesh filename")
            ("output-filename", po::value<string>(), "Output filename")
            ("mesh-output-filename", po::value<string>(), "Mesh output filename");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc),
                vm); // can throw

      /** --help option
       */
      if (vm.count("help"))
      {
        std::cout << "Basic Command Line Parameter App" << std::endl
                  << desc << std::endl;
        return 0;
      }

      po::notify(vm); // throws on error, so do after help in case
      // there are any problems
    }
    catch (po::error& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return -1;
    }

    const string mesh_filename = vm["mesh-filename"].as<string>();
    const FloatType voxel_size = vm["voxel-size"].as<FloatType>();
    boost::optional<string> output_filename = boost::none;
    boost::optional<string> mesh_output_filename = boost::none;
    if (vm.count("output-filename")) {
      output_filename = vm["output-filename"].as<string>();
    }
    if (vm.count("mesh-output-filename")) {
      mesh_output_filename = vm["mesh-output-filename"].as<string>();
    }
    run(mesh_filename, voxel_size, output_filename, mesh_output_filename);

    return 0;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
}
