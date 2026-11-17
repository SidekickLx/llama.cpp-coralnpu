// SPDX-License-Identifier: MIT
#include "tests/compiler/ggml_fixture.h"
#include "ggml-cpu.h"
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
void Require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void Fill(ggml_tensor *tensor, int seed) {
    auto *data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = static_cast<float>((i * 17 + seed) % 31 - 15) / 32.0f;
    }
    // Deliberately uninformative names: semantics must come from edges.
    ggml_set_name(tensor, "diagnostic-only");
}

void ResizeMetadata(ggml_tensor *tensor, int64_t columns, int64_t rows) {
    tensor->ne[0] = columns;
    tensor->ne[1] = rows;
    tensor->nb[1] = columns * sizeof(float);
    tensor->nb[2] = tensor->nb[3] = rows * tensor->nb[1];
}

struct Fixture {
    Context context{ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free};
    std::vector<const ggml_tensor *> inputs;
    std::vector<const ggml_tensor *> operations;
    std::vector<const ggml_tensor *> outputs;

    void Matmul(const std::string &name) {
        Require(context != nullptr, "GGML context creation failed");
        int64_t m = 2, k = 3, n = 4;
        if (name == "matmul-decode") { m = 1; k = 64; n = 172; }
        if (name == "matmul-prefill") { m = 5; k = 64; n = 172; }
        auto *x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, m);
        auto *w = name == "matmul-shared" ? x :
            ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, k, n);
        Fill(x, 4);
        if (w != x) {
            Fill(w, 11);
        }
        auto *y = ggml_mul_mat(context.get(), w, x);
        if (name == "matmul-explicit-f32") {
            Require(ggml_prec_set_acc(y, GGML_PREC_F32), "cannot set accumulator precision");
            Require(ggml_prec_set_src(y, GGML_PREC_F32, 1), "cannot set source precision");
        }
        inputs.push_back(x);
        if (w != x) {
            inputs.push_back(w);
        }
        operations = {y};
        outputs = {y};
        auto *graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, y);
        Require(ggml_graph_compute_with_ctx(context.get(), graph, 1) == GGML_STATUS_SUCCESS,
                "GGML CPU computation failed");

        // Malformed metadata is created only AFTER valid CPU execution.
        // Never pass invalid graphs to GGML kernels, which may assert.
        if (name == "bad-dtype-input") { x->type = GGML_TYPE_F16; }
        if (name == "bad-dtype-output") { y->type = GGML_TYPE_BF16; }
        if (name == "bad-quantized") { w->type = GGML_TYPE_Q4_0; }
        if (name == "bad-stride") { x->nb[1] += sizeof(float); }
        if (name == "bad-view") { x->view_src = w; }
        if (name == "bad-view-offset") { y->view_offs = 4; }
        if (name == "bad-extra") { x->extra = context.get(); }
        if (name == "bad-batch") { x->ne[2] = 2; }
        if (name == "bad-zero") { y->ne[0] = 0; }
        if (name == "bad-overflow") { x->ne[0] = INT64_MAX; }
        if (name == "bad-contracting") { ResizeMetadata(w, k + 1, n); }
        if (name == "bad-result-shape") { ResizeMetadata(y, n + 1, m); }
        if (name == "bad-missing-source") { y->src[1] = nullptr; }
        if (name == "bad-extra-source") { y->src[2] = x; }
        if (name == "bad-operator") { y->op = GGML_OP_ADD; }
        if (name == "bad-dependency") { inputs.pop_back(); }
        if (name == "bad-precision") { y->op_params[3] = GGML_PREC_F16; }
        if (name.rfind("bad-param-", 0) == 0) {
            const int word = std::stoi(name.substr(10));
            Require(word >= 0 && word < GGML_MAX_OP_PARAMS / 4, "invalid parameter word");
            y->op_params[word] = 12345;
        }
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error("Usage: coral-ggml-import-test CHECK_LIBRARY CASE NEW_OUTPUT_DIR");
        }
        const std::filesystem::path output(argv[3]);
        if (!output.parent_path().empty()) {
            std::filesystem::create_directories(output.parent_path());
        }
        Require(std::filesystem::create_directory(output), "output directory must be new");
        void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            throw std::runtime_error(dlerror());
        }
        auto check = reinterpret_cast<CoralCheckGGMLImport>(dlsym(library, "coral_check_ggml_import"));
        Require(check != nullptr, "compiler check library has no test entry point");
        Fixture fixture;
        fixture.Matmul(argv[2]);
        const CoralGGMLFixture view{
            1, argv[2], fixture.inputs.data(), fixture.inputs.size(),
            fixture.operations.data(), fixture.operations.size(),
            fixture.outputs.data(), fixture.outputs.size()};
        // Keep the library until process exit to preserve LLVM globals.
        return check(&view, argv[3]);
    } catch (const std::exception &error) {
        std::cerr << "coral-ggml-import-test: " << error.what() << '\n';
        return 1;
    }
}
