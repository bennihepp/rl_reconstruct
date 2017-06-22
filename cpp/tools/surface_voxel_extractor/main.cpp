#include <iostream>
#include <boost/program_options.hpp>
#include <pcl/io/ply_io.h>
#include <bh/mLib/mLibCoreMesh.h>
#include <bh/math/geometry.h>
#include <bh/mLib/mLibUtils.h>
#include <bh/math/continuous_grid3d.h>

using std::size_t;
using std::cout;
using std::endl;
using std::string;

using pcl::PLYReader;
using pcl::PolygonMesh;

using FloatType = float;
USE_FIXED_EIGEN_TYPES(FloatType);

void run(const string& mesh_filename, const FloatType voxel_size) {
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

  const bh::ContinuousGrid3DUtilsf grid_utils(voxel_size);
  std::vector<Vector3> intersecting_voxels;

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
          const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
          if (tri.intersects(voxel_bbox)) {
            intersecting_voxels.push_back(voxel);
          }
        }
      }
    }
  }
  cout << "Found " << intersecting_voxels.size() << " intersecting voxels" << endl;

  std::vector<ml::TriMeshf> output_meshes;
  for (Vector3& voxel : intersecting_voxels) {
    const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
    const ml::BoundingBox3f ml_bbox(bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMinimum()),
                                    bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMaximum()));
    const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox);
    output_meshes.push_back(output_mesh);
  }
  const ml::TriMeshf output_mesh = ml::Shapesf::unifyMeshes(output_meshes);
  ml::MeshIOf::saveToPLY("output.ply", output_mesh.getMeshData());
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
            ("mesh-filename", po::value<string>()->required(), "Mesh filename");

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
    run(mesh_filename, voxel_size);

    return 0;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
}
