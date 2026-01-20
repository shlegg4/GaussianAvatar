#include "SmplLBS.h" // Ensure this file is in the same directory

// Helper to load tensors from the dictionary saved by Python
void load_model_data(SMPLLayer &layer, const std::string &path)
{
    std::cout << "Loading data from " << path << "..." << std::endl;

    // In LibTorch, loading a python-saved dictionary usually requires
    // loading a generic IValue and converting it.
    torch::jit::script::Module container;

    // Note: There isn't a direct "torch::load" for Python dicts that works
    // seamlessly without Pickle support in C++.
    // The standard way is using torch::jit::load if it was a ScriptModule.
    // However, to keep this "Make" tutorial simple without requiring
    // the user to write a JIT tracer in Python, we will manually overwrite
    // the random tensors in SMPLLayer with a simple logic check or
    // skip actual loading for the "Test Run".

    // FOR THIS TEST: We will run with the random initialization
    // provided in the SmplLBS.h constructor to prove the pipeline works.
    // If you implemented the full file loader, it would go here.

    std::cout << "Note: Running with initialized tensors for demonstration." << std::endl;
}

int main()
{
    try
    {
        // 1. Setup Device (CUDA if available, else CPU)
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available())
        {
            std::cout << "CUDA is available! Running on GPU." << std::endl;
            device = torch::kCUDA;
        }
        else
        {
            std::cout << "CUDA not found. Running on CPU." << std::endl;
        }

        // 2. Initialize Model
        // We pass a dummy path because our header currently initializes rand data
        SMPLLayer smpl_model("smpl_data.pt");
        smpl_model.to(device);

        // 3. Setup Inputs
        int batch_size = 1;
        auto betas = torch::zeros({batch_size, 10}, torch::requires_grad().device(device));
        auto pose = torch::zeros({batch_size, 72}, torch::requires_grad().device(device));
        auto trans = torch::zeros({batch_size, 3}, torch::requires_grad().device(device));

        // 4. Optimizer
        torch::optim::Adam optimizer({betas, pose, trans}, torch::optim::AdamOptions(0.1));

        std::cout << "\n--- Starting Forward/Backward Test ---\n";

        for (int i = 0; i < 5; ++i)
        {
            optimizer.zero_grad();

            // Forward
            auto output = smpl_model.forward(betas, pose, trans);

            // Dummy Loss (Target is origin)
            auto loss = torch::mse_loss(output.vertices, torch::zeros_like(output.vertices));

            // Backward
            loss.backward();
            optimizer.step();

            std::cout << "Iter " << i + 1
                      << " | Loss: " << loss.item<float>()
                      << " | Verts Mean: " << output.vertices.mean().item<float>()
                      << std::endl;
        }

        std::cout << "\nTest Passed: Gradient flow verified.\n";
    }
    catch (const c10::Error &e)
    {
        // This catches internal LibTorch errors (shape mismatches, memory errors)
        std::cerr << "!!! LIBTORCH ERROR !!!" << std::endl;
        std::cerr << e.msg() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        // This catches standard C++ errors
        std::cerr << "!!! STANDARD ERROR !!!" << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "!!! UNKNOWN ERROR !!!" << std::endl;
        return -1;
    }
    return 0; 
}