#include <random>
#include <gtest/gtest.h>

#include "compute_graph.cuh"
#include "network.cuh"

#include "compute_graph_l1loss.cuh"
#include "operators.cuh"

#include "operator_elementwise.cuh"

#include "tools_cuda.cuh"


TEST(network, smoke) {
    ComputeGraph test_graph;
    test_graph._input_tensors.push_back(new Tensor());

    test_graph._weight_tensors.push_back(new Tensor({2, 2}));
    for (int i = 0; i < test_graph._weight_tensors[0]->_element_count; i++) {
        test_graph._weight_tensors[0]->_p_data[i] = i;
    }
    Operator *ele = new ElementWise(test_graph._input_tensors[0], test_graph._weight_tensors[0], ELE_OP::ADD);

    Network test_net(&test_graph, true, cudaStreamDefault);
    test_net.to(cudaMemoryTypeDevice);

    std::vector<Tensor*> sample_inputs{new Tensor({2, 2})};
    sample_inputs[0]->fill_data_random(0.9, 1.0);
    test_net.init(sample_inputs, "");

    test_net._weight_tensors[0]->update_weights(1.f, cudaStreamDefault);
    for (int i = 0; i < 2; i++){
        Tensor t1({2, 2});
        t1.malloc_gradient();
        for (int i = 0; i < t1._element_count; i++) {
            t1._p_data[i] = i/10.f;
        }
        auto outs = test_net.forward({&t1});
        for (auto o: outs) {
            Tensor o_h(*o);
            o_h.to(cudaMemoryTypeHost);
            std::cout << o_h;
        }
        checkCudaErrors(cudaMemcpy(test_net.get_output_tensors()[0]->_p_gradient, sample_inputs[0]->_p_data, 16, cudaMemcpyHostToDevice));
        
        test_net.backward();
        test_net.update_weights(0.5);
        Tensor w(*test_net._weight_tensors[0]);
        w.to(cudaMemoryTypeHost);
        std::cout << "weight: " << w << std::endl;


        std::cout << "---------------" << std::endl;
    }
}

TEST(network, mm) {
    ComputeGraph *mm_graph = new ComputeGraph();
    mm_graph->_input_tensors.push_back(new Tensor());
    mm_graph->_weight_tensors.push_back(new Tensor({2, 2}));
    new MatMul(mm_graph->_weight_tensors[0], mm_graph->_input_tensors[0]);

    ComputeGraph *l1loss_graph = new L1LossGraph();


    Tensor *input = new Tensor({2, 1});

    Tensor *target = new Tensor({2, 1});


    Network mm_net(mm_graph, true, cudaStreamDefault);
    mm_net.to(cudaMemoryTypeDevice);
    mm_net._weight_tensors[0]->fill_data_random(-1.0, 1.0);
    mm_net.init({input}, "");


    Network l1loss(l1loss_graph, true, cudaStreamDefault);
    l1loss.to(cudaMemoryTypeDevice);


    for (int i = 0; i < 999; i++) {
        input->fill_data_random(-1.0, 1.0);
        target->_p_data[0] = input->_p_data[0] + input->_p_data[1];
        target->_p_data[1] = input->_p_data[0] - input->_p_data[1];
        std::vector<Tensor*> predict = mm_net.forward({input});
        std::vector<Tensor*> loss = l1loss.forward({predict[0], target});

        std::cout << "loss: " << loss[0] << std::endl;

        l1loss.backward();
        *mm_net.get_output_tensors()[0] = *l1loss._input_tensors[0];
        mm_net.backward();

        mm_net.update_weights(0.1);
    }

}