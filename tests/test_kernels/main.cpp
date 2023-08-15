#include<iostream>
#include<gtest/gtest.h>
#include <glog/logging.h>
#include "tools_common.cuh"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO);
    return RUN_ALL_TESTS();
}