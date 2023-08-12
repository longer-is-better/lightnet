clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build && cuda-gdb -ex "break test_conv2d.cu:196" --args build/tests/test_operators/test_operators --gtest_filter=test_conv2d.smoke
clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON && cmake --build build
clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=release -DCMAKE_VERBOSE_MAKEFILE=off && cmake --build build && build/tests/test_networks/test_networks --gtest_filter=alexnet.smoke
python3 tests/test_networks/test_alexnet/torch_alexnet_smoke.py

clear; cmake -S . -B build -DCMAKE_BUILD_TYPE=release -DCMAKE_VERBOSE_MAKEFILE=off && cmake --build build && pushd build; ctest > log 2>&1; pushd
build/tests/test_operators/test_operators --gtest_filter=cudnnConvolutionForward.smoke