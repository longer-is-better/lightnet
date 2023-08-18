#include <random>
#include <gtest/gtest.h>

#include "compute_graph.cuh"
#include "network.cuh"

#include "compute_graph_l1loss.cuh"
#include "operators.cuh"

#include "operator_elementwise.cuh"

#include "tools_cuda.cuh"
#include "tools_common.cuh"


TEST(network, mm) {
    ComputeGraph *mm_graph = new ComputeGraph();
    mm_graph->_input_tensors.push_back(new Tensor());
    mm_graph->_weight_tensors.push_back(new Tensor({2, 2}));



    // mm_graph->_weight_tensors[0]->_p_data[0] = 0.5f; // 1.f;
    // mm_graph->_weight_tensors[0]->_p_data[1] = 0.5f; // 1.f;
    // mm_graph->_weight_tensors[0]->_p_data[2] = 0.5f; // 1.f;
    // mm_graph->_weight_tensors[0]->_p_data[3] = 0.5f; // -1.f;







    new MatMul(mm_graph->_weight_tensors[0], mm_graph->_input_tensors[0]);

    ComputeGraph *l1loss_graph = new L1LossGraph();


    Tensor *input = new Tensor({2, 1});

    Tensor *target = new Tensor({2, 1});


    Network mm_net(mm_graph, cudaStreamDefault);
    mm_net.to(cudaMemoryTypeDevice);
    mm_net._weight_tensors[0]->fill_data_random(-1.0, 1.0);
    std::vector<Tensor*> init_out = mm_net.init({input}, "");


    Network l1loss(l1loss_graph, cudaStreamDefault);
    l1loss.to(cudaMemoryTypeDevice);
    l1loss.init({init_out[0], target}, "");

    for (int i = 0; i < 2; i++) {
        input->fill_data_random(-1.0, 1.0);
        // input->_p_data[0] = 2.f;
        // input->_p_data[1] = -1.f;
        // target->_p_data[0] = 1 * input->_p_data[0];
        target->_p_data[0] = input->_p_data[0] + input->_p_data[1];
        target->_p_data[1] = input->_p_data[0] - input->_p_data[1];

        std::cout << "input" << *input;
        std::cout << "weight" << *mm_net._weight_tensors[0];
    
        std::vector<Tensor*> predict = mm_net.forward({input});
        std::cout << "target" << *target;
        std::cout << "predict" << *predict[0];

        std::vector<Tensor*> loss = l1loss.forward({predict[0], target});

        Tensor show_loss;
        show_loss = *loss[0];
        std::cout << "loss: " << show_loss << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;

        l1loss.backward();
        *mm_net.get_output_tensors()[0] = *l1loss._input_tensors[0];
        mm_net.backward();

        mm_net.update_weights(0.01);

        Tensor show_weight_grad;
        show_weight_grad = *mm_net._weight_tensors[0];
        memcpy(show_weight_grad._p_data, show_weight_grad._p_gradient, show_weight_grad._total_size);
        std::cout << "weight grad" << show_weight_grad;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }

}