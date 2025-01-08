// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/version.h>  // For TORCH_VERSION* macros
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "matmul.h"

// Copied from https://github.com/pytorch/pytorch/commit/7931eee5c5ebcdf468bff4d308510b03355cd909
// This is so that we can pass in torch.dtype as a parameter to the function.
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11::detail {

    template <>
    struct type_caster<at::ScalarType> {
    public:
        // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
        PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));
        // PYBIND11_TYPE_CASTER defines a member field called value. at::ScalarType
        // cannot be default-initialized, we provide this constructor to explicitly
        // initialize that field. The value doesn't matter as it will be overwritten
        // after a successful call to load.
        type_caster() : value(at::kFloat) {}
        bool load(handle src, bool) {
            PyObject* obj = src.ptr();
            if (THPDtype_Check(obj)) {
                value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
                return true;
            }
            return false;
        }
        static handle cast(
                           const at::ScalarType& src,
                           return_value_policy /* policy */,
                           handle /* parent */) {
            return Py_NewRef(torch::getTHPDtype(src));
        }
    };

} // namespace pybind11::detail

#endif

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_mmprop(MM_kernel_params &params,
                      // sizes
                      const size_t m,
                      const size_t n,
                      const size_t k,

                      // device pointers
                      const at::Tensor a,
                      const at::Tensor b,
                      at::Tensor c,
                      const int k_id=0,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.A_ptr = a.data_ptr();
    params.B_ptr = b.data_ptr();
    params.C_ptr = c.data_ptr();

    // Set the dimensions.
    params.M = m;
    params.N = n;
    params.K = k

    params.kernel_id = k_id;

    params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
}




// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::vector<at::Tensor>
mm_fwd(at::Tensor &a,   // (m, n)
        const at::Tensor &b,  // (n, k) 
        c10::optional<at::Tensor> &c_,  // (m, k)
        int kernel_id, // kernel_id
        int const sm_margin
        ) {

    int m = a.size(0);
    int n = a.size(1);
    int k = b.size(1);
    
    auto opts = q.options();
    auto q_type = q.scalar_type();
    at::Tensor out;
    if (c_.has_value()) {
        c = c_.value();
    } else {
        c = torch::empty_like((m, k), opts.dtype(q_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    MM_kernel_params params;
    set_params_mmprop(params, m, n, k, a, b, c, kernel_id, sm_margin);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_matmul(params, stream);

    // return {out, softmax_lse};
    return {c};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Matmul";
    m.def("fwd", &mm_fwd, "Forward pass");
}

