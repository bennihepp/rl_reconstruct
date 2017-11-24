#include <iostream>
#include <unordered_set>
#include <stack>
#include <boost/program_options.hpp>
#include <boost/optional.hpp>
//#include <pcl/io/ply_io.h>
#include <bh/eigen_utils.h>
#include <bh/eigen_options.h>
#include <bh/mLib/mLibCoreMesh.h>
#include <bh/math/geometry.h>
#include <bh/mLib/mLibUtils.h>
#include <bh/math/continuous_grid3d.h>
#include <bh/mesh/triangle_mesh.h>

using std::size_t;
using std::cout;
using std::endl;
using std::string;

//using pcl::PLYReader;
//using pcl::PolygonMesh;

using FloatType = float;
USE_FIXED_EIGEN_TYPES(FloatType);
using BoundingBoxType = bh::BoundingBox3D<FloatType>;
using TriangleType = bh::Triangle<FloatType>;
using TriangleMeshType = bh::TriangleMesh<FloatType>;
using TriangleMeshFactoryType = bh::TriangleMeshFactory<FloatType>;

void run(const string& mesh_filename, const FloatType voxel_size,
         const boost::optional<string>& text_output_filename,
         const boost::optional<string>& binary_output_filename,
         const boost::optional<string>& mesh_output_filename,
         const BoundingBoxType& clip_bbox,
         const FloatType max_squared_triangle_area,
         const bool debug_output) {
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

  std::stack<TriangleType> triangle_stack;
  for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); ++i) {
    const ml::MeshDataf::Indices::Face& face = mesh.m_FaceIndicesVertices[i];
    BH_ASSERT_STR(face.size() == 3, "Can only handle triangle meshes");
    const Vector3 v1 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[0]]);
    const Vector3 v2 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[1]]);
    const Vector3 v3 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[2]]);
    if (clip_bbox.isOutside(v1) && clip_bbox.isOutside(v2) && clip_bbox.isOutside(v3)) {
      const TriangleType tri(v1, v2, v3);
      if (!tri.intersects(clip_bbox)) {
        continue;
      }
    }
    const TriangleType tri(v1, v2, v3);
    triangle_stack.push(tri);
  }

  // For debugging
  if (debug_output) {
    BoundingBoxType mesh_bbox = BoundingBoxType::Empty();
    for (size_t i = 0; i < mesh.m_FaceIndicesVertices.size(); ++i) {
      const ml::MeshDataf::Indices::Face& face = mesh.m_FaceIndicesVertices[i];
      BH_ASSERT_STR(face.size() == 3, "Can only handle triangle meshes");
      const Vector3 v1 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[0]]);
      const Vector3 v2 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[1]]);
      const Vector3 v3 = bh::MLibUtilities::convertMlibToEigen(mesh.m_Vertices[face[2]]);
      mesh_bbox.include(v1);
      mesh_bbox.include(v2);
      mesh_bbox.include(v2);
    }
    std::cout << "Mesh bounding box: " << mesh_bbox << std::endl;
  }

  // For debugging
  if (debug_output) {
    std::vector<TriangleType> triangles;
    while (!triangle_stack.empty()) {
      triangles.push_back(triangle_stack.top());
      triangle_stack.pop();
    }
    TriangleMeshType tri_mesh = TriangleMeshFactoryType::createFromTriangles(triangles.begin(), triangles.end());
    ml::MeshDataf tri_mesh_ml = bh::MLibUtilities::convertBhToMlib(tri_mesh);
    ml::MeshIOf::saveToPLY("tmp_clip_mesh.ply", tri_mesh_ml);
    for (const TriangleType& triangle : triangles) {
      triangle_stack.push(triangle);
    }
  }
  
  std::cout << "Remaining triangles after clipping: " << triangle_stack.size() << std::endl;

  std::vector<TriangleType> triangles;
  cout << "Splitting up triangles" << endl;
  size_t pushed_tri_count = triangle_stack.size();
  size_t processed_tri_count = 0;
  while (!triangle_stack.empty()) {
    const TriangleType tri = triangle_stack.top();
    triangle_stack.pop();
    const float area_squared = tri.computeTriangleAreaSquare();
    if (area_squared > max_squared_triangle_area) {
//      const std::array<TriangleType, 2> split_tris = tri.splitTriangleInto2();
      const std::array<TriangleType, 3> split_tris = tri.splitTriangleInto3();
      for (size_t i = 0; i < split_tris.size(); ++i) {
        triangle_stack.push(split_tris[i]);
      }
      pushed_tri_count += split_tris.size();
      if (pushed_tri_count % 1000 == 0) {
        cout << "Pushed " << pushed_tri_count << " triangles" << endl;
      }
    }
    else {
      triangles.push_back(tri);
      ++processed_tri_count;
      if (processed_tri_count % 1000 == 0) {
        cout << "Processed " << processed_tri_count << " triangles" << endl;
      }
    }
  }

  // For debugging
  if (debug_output) {
    TriangleMeshType tri_mesh = TriangleMeshFactoryType::createFromTriangles(triangles.begin(), triangles.end());
    ml::MeshDataf tri_mesh_ml = bh::MLibUtilities::convertBhToMlib(tri_mesh);
    ml::MeshIOf::saveToPLY("tmp_split_mesh.ply", tri_mesh_ml);
  }

  // For debugging
  std::vector<ml::TriMeshf> tmp_voxel_meshes;
  std::vector<ml::TriMeshf> tmp_intersecting_voxel_meshes;

  cout << "Intersecting " << triangles.size() << " triangles with grid voxels" << endl;
  // Iterate over all triangles
//#pragma omp parallel for
  for (size_t i = 0; i < triangles.size(); ++i) {
    if (i % 100 == 0) {
      cout << "Processing triangle " << i + 1 << " out of " << triangles.size() << endl;
    }
    const TriangleType& tri = triangles[i];

    // Loop over all voxels in triangle bounding box
    const bh::BoundingBox3Df bbox = tri.boundingBox();
    const Vector3 min_voxel = grid_utils.getMinIntersectingVoxel(bbox);
    const Vector3 max_voxel = grid_utils.getMaxIntersectingVoxel(bbox);
    if (debug_output) {
      std::cout << "i: " << i << std::endl;
      std::cout << "v1: " << tri.v1().transpose() << std::endl;
      std::cout << "v2: " << tri.v2().transpose() << std::endl;
      std::cout << "v3: " << tri.v3().transpose() << std::endl;
      std::cout << "bbox: " << bbox << std::endl;
      std::cout << "min_voxel: " << min_voxel.transpose() << std::endl;
      std::cout << "max_voxel: " << max_voxel.transpose() << std::endl;
    }
    for (FloatType x = min_voxel(0); x <= max_voxel(0); x += grid_utils.getIncrement()) {
      for (FloatType y = min_voxel(1); y <= max_voxel(1); y += grid_utils.getIncrement()) {
        for (FloatType z = min_voxel(2); z <= max_voxel(2); z += grid_utils.getIncrement()) {
          const Vector3 voxel(x, y, z);
          const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
          // For debugging
          if (debug_output) {
            const ml::BoundingBox3f ml_bbox(bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMinimum()),
                                            bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMaximum()));
            const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox);
            tmp_voxel_meshes.push_back(output_mesh);
          }
          if (tri.intersects(voxel_bbox)) {
            // For debugging
            if (debug_output) {
              const ml::BoundingBox3f ml_bbox(bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMinimum()),
                                              bh::MLibUtilities::convertEigenToMlib(voxel_bbox.getMaximum()));
              const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox);
              tmp_intersecting_voxel_meshes.push_back(output_mesh);
            }
          }
          if (!clip_bbox.intersects(voxel_bbox)) {
            continue;
          }
          // This does not seem to do any speedup and prevents parallelization.
//          if (intersecting_voxels.find(voxel) != intersecting_voxels.end()) {
//            continue;
//          }
          if (tri.intersects(voxel_bbox)) {
//#pragma omp critical
            {
              intersecting_voxels.insert(voxel);
            }
          }
        }
      }
    }
  }
  cout << "Found " << intersecting_voxels.size() << " intersecting voxels" << endl;

  // For debugging
  if (debug_output) {
    const ml::TriMeshf tmp_voxel_mesh = ml::Shapesf::unifyMeshes(tmp_voxel_meshes);
    ml::MeshIOf::saveToPLY("tmp_voxel_mesh.ply", tmp_voxel_mesh.getMeshData());
    const ml::TriMeshf tmp_intersecting_voxel_mesh = ml::Shapesf::unifyMeshes(tmp_intersecting_voxel_meshes);
    ml::MeshIOf::saveToPLY("tmp_intersecting_voxel_mesh.ply", tmp_intersecting_voxel_mesh.getMeshData());
  }

  Vector3 surface_min(std::numeric_limits<FloatType>::max(),
                      std::numeric_limits<FloatType>::max(),
                      std::numeric_limits<FloatType>::max());
  Vector3 surface_max(- std::numeric_limits<FloatType>::max(),
                      - std::numeric_limits<FloatType>::max(),
                      - std::numeric_limits<FloatType>::max());
  for (const Vector3& voxel : intersecting_voxels) {
    for (std::size_t i = 0; i < 3; ++i) {
      if (voxel(i) < surface_min(i)) {
        surface_min(i) = voxel(i);
      }
      if (voxel(i) > surface_max(i)) {
        surface_max(i) = voxel(i);
      }
    }
  }
  bh::BoundingBox3Df surface_bbox(surface_min, surface_max);
  cout << "Surface bounding box: " << surface_bbox << endl;

  if (text_output_filename) {
    cout << "Writing surface voxels to text file" << endl;
    std::ofstream out(*text_output_filename);
    if (!out) {
      throw std::runtime_error("ERROR: Unable to open text output file: " + *text_output_filename);
    }
    for (const Vector3 &voxel : intersecting_voxels) {
      out << voxel(0) << " " << voxel(1) << " " << voxel(2) << endl;
    }
  }

  if (binary_output_filename) {
    cout << "Writing surface voxels to binary file" << endl;
    std::ofstream out(*binary_output_filename, std::ios_base::binary);
    if (!out) {
      throw std::runtime_error("ERROR: Unable to open binary output file: " + *binary_output_filename);
    }
    const size_t num_voxels = intersecting_voxels.size();
    out.write(reinterpret_cast<const char*>(&num_voxels), sizeof(num_voxels));
    for (const Vector3 &voxel : intersecting_voxels) {
      out.write(reinterpret_cast<const char*>(&voxel(0)), sizeof(voxel(0)));
      out.write(reinterpret_cast<const char*>(&voxel(1)), sizeof(voxel(1)));
      out.write(reinterpret_cast<const char*>(&voxel(2)), sizeof(voxel(2)));
    }
  }

  if (mesh_output_filename) {
    cout << "Writing surface voxels to mesh" << endl;
    std::vector<ml::TriMeshf> output_meshes;
    for (const Vector3 &voxel : intersecting_voxels) {
      const bh::BoundingBox3Df voxel_bbox(voxel, grid_utils.getIncrement());
      const ml::BoundingBox3f ml_bbox = bh::MLibUtilities::convertBhToMlib(voxel_bbox);
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
            ("text-output-filename", po::value<string>(), "Text output filename")
            ("binary-output-filename", po::value<string>(), "Binary output filename")
            ("mesh-output-filename", po::value<string>(), "Mesh output filename")
            ("bbox-min", po::value<Vector3>(), "Bounding box minimum")
            ("bbox-max", po::value<Vector3>(), "Bounding box maximum")
            ("max-triangle-area", po::value<FloatType>()->default_value(5.0f), "Max triangle area")
            ("debug-output", po::bool_switch()->default_value(false), "Produce debug output");

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
    boost::optional<string> text_output_filename = boost::none;
    boost::optional<string> binary_output_filename = boost::none;
    boost::optional<string> mesh_output_filename = boost::none;
    if (vm.count("text_output-filename")) {
      text_output_filename = vm["text-output-filename"].as<string>();
    }
    if (vm.count("binary-output-filename")) {
      binary_output_filename = vm["binary-output-filename"].as<string>();
    }
    if (vm.count("mesh-output-filename")) {
      mesh_output_filename = vm["mesh-output-filename"].as<string>();
    }

    const FloatType lowest = std::numeric_limits<FloatType>().lowest();
    const FloatType highest = std::numeric_limits<FloatType>().max();
    Vector3 bbox_min(lowest, lowest, lowest);
    Vector3 bbox_max(highest, highest, highest);
    if (vm.count("bbox-min")) {
      bbox_min = vm["bbox-min"].as<Vector3>();
    }
    if (vm.count("bbox-max")) {
      bbox_max = vm["bbox-max"].as<Vector3>();
    }
    std::cout << "Minimum bounding box: " << bbox_min.transpose() << std::endl;
    std::cout << "Maximum bounding box: " << bbox_max.transpose() << std::endl;
    const BoundingBoxType bbox(bbox_min, bbox_max);
    const FloatType max_triangle_area = vm["max-triangle-area"].as<FloatType>();
    const FloatType max_squared_triangle_area = max_triangle_area * max_triangle_area;
    run(mesh_filename, voxel_size,
        text_output_filename, binary_output_filename, mesh_output_filename,
        bbox, max_squared_triangle_area, vm["debug-output"].as<bool>());

    return 0;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
}
