//==================================================
// convert_bbox_to_mesh.cpp
//
//  Copyright (c) 2017 Benjamin Hepp.
//  Author: Benjamin Hepp
//  Created on: 23.10.17
//==================================================

#include <iostream>
#include <boost/program_options.hpp>
#include <bh/eigen_utils.h>
#include <bh/mLib/mLibCoreMesh.h>
#include <bh/math/geometry.h>
#include <bh/mLib/mLibUtils.h>

using std::cout;
using std::endl;
using std::string;

using FloatType = float;
USE_FIXED_EIGEN_TYPES(FloatType);
using BoundingBoxType = bh::BoundingBox3D<FloatType>;
using ColorType = bh::Color4<FloatType>;

void run(const BoundingBoxType& bbox,
         const ColorType& color,
         const string& mesh_output_filename) {
  std::vector<ml::TriMeshf> output_meshes;
  const ml::BoundingBox3f ml_bbox = bh::MLibUtilities::convertBhToMlib(bbox);
  const ml::vec4<FloatType> ml_color = bh::MLibUtilities::convertEigenToMlib(color);
  const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox, ml_color);
  ml::MeshIOf::saveToPLY(mesh_output_filename, output_mesh.getMeshData());
}

int main(int argc, const char** argv) {
  try {
    /** Define and parse the program options
     */
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
            ("help", "Print help messages")
            ("mesh-output-filename", po::value<string>()->required(), "Mesh output filename")
            ("color", po::value<ColorType>()->required(), "Color of bbox")
            ("bbox", po::value<BoundingBoxType >()->required(), "Bounding box");

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

      po::notify(vm); // throws on error, so do after help in case there are any problems
    }
    catch (po::error& e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return -1;
    }

    run(vm["bbox"].as<BoundingBoxType>(),
        vm["color"].as<ColorType>(),
        vm["mesh-output-filename"].as<string>());

    return 0;
  }
  catch (std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return -1;
  }
}
