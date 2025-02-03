from torch.utils.cpp_extension import load
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

import torch


# HACK: we monkey patch pytorch's _write_ninja_file to pass
# "-gencode arch=compute_sm90a,code=sm_90a" to files ending in '_sm90.cu',
# and pass "-gencode arch=compute_sm80,code=sm_80" to files ending in '_sm80.cu'
from torch.utils.cpp_extension import (
    IS_HIP_EXTENSION,
    COMMON_HIP_FLAGS,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    get_cxx_compiler,
    _join_rocm_home,
    _join_cuda_home,
    _is_cuda_file,
    _maybe_write,
)

def _write_ninja_file(path,
                      cflags,
                      post_cflags,
                      cuda_cflags,
                      cuda_post_cflags,
                      cuda_dlink_post_cflags,
                      sources,
                      objects,
                      ldflags,
                      library_target,
                      with_cuda) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """
    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        nvcc_from_env = 'nvcc'
        if "PYTORCH_NVCC" in os.environ:
            nvcc_from_env = os.getenv("PYTORCH_NVCC")    # user can set nvcc compiler with ccache using the environment variable here
        if IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc_from_env = {nvcc_from_env}')
        config.append(f'nvcc = {nvcc}')

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_cflags_sm90a = ['-gencode=arch=compute_90a,code=sm_90a' if s == '-gencode=arch=compute_90,code=sm_90' else s for s in cuda_cflags]
        flags.append(f'cuda_cflags_sm90a = {" ".join(cuda_cflags_sm90a)}')
        cuda_post_cflags_sm80 = [s if s != 'arch=compute_90a,code=sm_90a' else 'arch=compute_80,code=sm_80' for s in cuda_post_cflags]
        flags.append(f'cuda_post_cflags_sm80 = {" ".join(cuda_post_cflags_sm80)}')
        cuda_post_cflags_sm80_sm90 = cuda_post_cflags + ['-gencode', 'arch=compute_80,code=sm_80']
        flags.append(f'cuda_post_cflags_sm80_sm90 = {" ".join(cuda_post_cflags_sm80_sm90)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append(
            '  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append(
            '  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')

    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        # --generate-dependencies-with-compile is not supported by ROCm
        # Nvcc flag `--generate-dependencies-with-compile` is not supported by sccache, which may increase build time.
        if torch.version.cuda is not None and os.getenv('TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES', '0') != '1':
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            # Note: non-system deps with nvcc are only supported
            # on Linux so use --generate-dependencies-with-compile
            # to make this work on Windows too.
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule_sm80 = ['rule cuda_compile_sm80'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80'
        ]
        cuda_compile_rule_sm80_sm90 = ['rule cuda_compile_sm80_sm90'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_sm80_sm90'
        ]
        cuda_compile_rule_sm90a = ['rule cuda_compile_sm90a'] + cuda_compile_rule[1:] + [
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags_sm90a -c $in -o $out $cuda_post_cflags'
        ]
        cuda_compile_rule.append(
            f'  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            rule = 'cuda_compile_sm90a'
        else:
            rule = 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f'build {object_file}: {rule} {source_file}')

    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f'build {devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where',
                                                'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f'default {library_target}']
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm80_sm90)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_sm90a)
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


# Monkey patching
torch.utils.cpp_extension._write_ninja_file = _write_ninja_file

matmul_cuda = load(
    name='matmul_cuda', 
    sources=['matmul_api.cpp', 'matmul.cu'],
    extra_include_paths=['examples/matmul'],
    extra_cuda_cflags=['-std=c++17 -O3 -DNDEBUG -w --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -lineinfo -g', 
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",],
    extra_ldflags=['-Wl,--no-as-needed', '-lm', '-lcublas', '-lcuda'],
    with_cuda=True,
    verbose=True,
    )

import torch.profiler as profiler
from torch.profiler import ProfilerActivity


M, N, K = 8192, 8192, 8192
torch.manual_seed(0)
a = torch.rand(M, N, dtype=torch.bfloat16).cuda()
b = torch.rand(N, K, dtype=torch.bfloat16).cuda()
# print("a", a)
# print("b", b)

c = torch.zeros(M, K, dtype=torch.bfloat16).cuda()

with profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    with_stack=True, 
    ) as prof:

    torch.profiler.itt.range_push("Forward Tiled Matmul")
    c = matmul_cuda.fwd(a, b, c, 1, 0)
    torch.profiler.itt.range_pop()
    torch.profiler.itt.range_push("PyTorch Matmul")
    c_ref = torch.matmul(a, b)
    torch.profiler.itt.range_pop()
    
prof.export_chrome_trace("trace.json")
print(c)
print("a@b", c_ref)
torch.testing.assert_close(c, c_ref)


from typing import Callable
from torch._inductor.runtime.benchmarking import benchmarker
def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs)) * 1e3


for kernel_id in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}:
    forward_time = benchmark_torch_function_in_microseconds(matmul_cuda.fwd, a, b, c, kernel_id, 0)
    print(f"KERNEL {kernel_id}: {forward_time:.2f} us")