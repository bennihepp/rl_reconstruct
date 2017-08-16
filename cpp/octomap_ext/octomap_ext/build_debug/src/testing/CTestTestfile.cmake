# CMake generated Testfile for 
# Source directory: /home/t-behepp/src/octomap/octomap/src/testing
# Build directory: /home/t-behepp/src/octomap/octomap/build_debug/src/testing
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(MathVector "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "MathVector")
add_test(MathPose "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "MathPose")
add_test(InsertRay "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "InsertRay")
add_test(InsertScan "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "InsertScan")
add_test(ReadGraph "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "ReadGraph")
add_test(StampedTree "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "StampedTree")
add_test(OcTreeKey "/home/t-behepp/src/octomap/octomap/bin/unit_tests" "OcTreeKey")
add_test(test_scans "/home/t-behepp/src/octomap/octomap/bin/test_scans" "/home/t-behepp/src/octomap/octomap/share/data/spherical_scan.graph")
add_test(test_raycasting "/home/t-behepp/src/octomap/octomap/bin/test_raycasting")
add_test(test_io "/home/t-behepp/src/octomap/octomap/bin/test_io" "/home/t-behepp/src/octomap/octomap/share/data/geb079.bt")
add_test(test_pruning "/home/t-behepp/src/octomap/octomap/bin/test_pruning")
add_test(test_iterators "/home/t-behepp/src/octomap/octomap/bin/test_iterators" "/home/t-behepp/src/octomap/octomap/share/data/geb079.bt")
add_test(test_mapcollection "/home/t-behepp/src/octomap/octomap/bin/test_mapcollection" "/home/t-behepp/src/octomap/octomap/share/data/mapcoll.txt")
add_test(test_color_tree "/home/t-behepp/src/octomap/octomap/bin/test_color_tree")
