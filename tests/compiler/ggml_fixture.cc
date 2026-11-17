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

    void Compute(ggml_tensor *result) {
        auto *graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, result);
        Require(ggml_graph_compute_with_ctx(context.get(), graph, 1) == GGML_STATUS_SUCCESS,
                "GGML CPU computation failed");
    }

    void Pointwise(const std::string &name) {
        const bool unary = name.rfind("silu", 0) == 0 || name.rfind("bad-silu", 0) == 0;
        const bool multiply = name.rfind("mul-", 0) == 0 || name.rfind("bad-mul-", 0) == 0;
        const bool packed = name == "bad-glu-packed";
        auto *a = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, packed ? 18 : 9, 2);
        auto *b = ggml_new_tensor_2d(
            context.get(), GGML_TYPE_F32, 9, name == "bad-mul-broadcast" ? 1 : 2);
        const float activation[] = {-80.0f, -10.0f, -1.0f, -1e-7f, 0, 1e-7f, 1, 10, 80};
        auto *data = static_cast<float *>(a->data);
        for (int64_t i = 0; i < ggml_nelements(a); ++i) {
            data[i] = activation[i % 9];
        }
        Fill(b, 9);
        ggml_tensor *result = nullptr;
        if (unary) {
            result = name == "bad-silu-inplace" ?
                ggml_silu_inplace(context.get(), a) : ggml_silu(context.get(), a);
            inputs = {a};
        } else if (multiply) {
            result = ggml_mul(context.get(), a, b);
            inputs = {a, b};
        } else if (packed) {
            result = ggml_swiglu(context.get(), a);
            inputs = {a};
        } else {
            result = name == "glu-reversed" ?
                ggml_swiglu_split(context.get(), b, a) : ggml_swiglu_split(context.get(), a, b);
            inputs = {a, b};
        }
        operations = {result};
        outputs = {result};
        Compute(result);
        if (name == "bad-silu-activation") { result->op_params[0] = GGML_UNARY_OP_RELU; }
        if (name == "bad-silu-extra-source") {
            result->src[1] = b;
            inputs.push_back(b);
        }
        if (name == "bad-glu-swap") { result->op_params[1] = 1; }
        if (name == "bad-glu-clamp") { result->op_params[0] = GGML_GLU_OP_SWIGLU_CLAMP; }
        if (name == "bad-glu-oai") { result->op_params[0] = GGML_GLU_OP_SWIGLU_OAI; }
        if (name == "bad-glu-shape") { ResizeMetadata(b, 9, 1); }
        if (name == "bad-glu-missing-source") { result->src[1] = nullptr; }
        if (name == "bad-glu-dtype") { b->type = GGML_TYPE_F16; }
        if (name == "bad-mul-result-shape") { ResizeMetadata(result, 9, 1); }
        const auto parameter = name.find("-param-");
        if (parameter != std::string::npos) {
            const int word = std::stoi(name.substr(parameter + 7));
            Require(word >= 0 && word < GGML_MAX_OP_PARAMS / 4, "invalid parameter word");
            result->op_params[word] = 12345;
        }
    }

    void FFN(const std::string &name) {
        const bool region = name.find("region") != std::string::npos;
        const bool tail = name.find("tail") != std::string::npos;
        const bool shared = name.find("shared") != std::string::npos;
        const bool expanded = name.find("expanded") != std::string::npos || (!region && name != "ffn-split");
        int64_t m = tail ? 3 : 2;
        int64_t d = tail ? 7 : 3;
        int64_t f = tail ? 11 : 4;
        if (name.find("decode") != std::string::npos) {
            m = 1;
        } else if (name.find("prefill") != std::string::npos) {
            m = 5;
        }
        if (name.rfind("ffn-region-profile-m", 0) == 0) {
            m = std::stoll(name.substr(std::string("ffn-region-profile-m").size()));
            Require(m == 1 || m == 16 || m == 32, "unknown contract profile");
            d = 128;
            f = 256;
        }
        auto *x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, d, m);
        auto *wg = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, d, f);
        auto *wu = shared ? wg : ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, d, f);
        auto *wd = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, f, d);
        Fill(x, 1);
        Fill(wg, 5);
        if (!shared) {
            Fill(wu, 17);
        }
        Fill(wd, 21);
        if (name == "ffn-region-gate-up-swapped") {
            std::swap(wg, wu);
        }
        auto *gate = ggml_mul_mat(context.get(), wg, x);
        auto *up = ggml_mul_mat(context.get(), wu, x);
        inputs = {x, wg, wu, wd};
        operations = {gate, up};
        outputs = {gate, up};
        ggml_tensor *hidden = nullptr;
        if (!expanded) {
            hidden = ggml_swiglu_split(context.get(), gate, up);
        } else {
            auto *silu = ggml_silu(context.get(), gate);
            hidden = name.find("reversed") != std::string::npos ?
                ggml_mul(context.get(), up, silu) : ggml_mul(context.get(), silu, up);
            operations.push_back(silu);
            outputs.push_back(silu);
        }
        auto *y = ggml_mul_mat(context.get(), wd, hidden);
        operations.push_back(hidden);
        operations.push_back(y);
        outputs.push_back(hidden);
        outputs.push_back(y);
        Compute(y);
        if (region) {
            outputs = {y};
        }
        if (name == "ffn-region-up-first") {
            std::swap(operations[0], operations[1]);
        }
        if (name == "bad-ffn-order" || name == "bad-region-order") {
            std::swap(operations[0], operations.back());
        }
        if (name == "bad-ffn-self-cycle" || name == "bad-region-cycle") {
            y->src[1] = y;
        }
        if (name == "bad-region-roles") {
            std::swap(inputs[1], inputs[2]);
        }
        if (name == "bad-region-missing") {
            operations.pop_back();
        }
        if (name == "bad-region-duplicate") {
            operations[1] = operations[0];
        }
        if (name == "bad-region-parameter") {
            hidden->op_params[1] = 1;
        }
        if (name == "bad-region-weight-producer") {
            wg->op = GGML_OP_ADD;
        }
        if (name == "bad-region-output") {
            outputs = {hidden};
        }
        if (name == "bad-region-payload-swap") {
            auto *gate_data = static_cast<float *>(wg->data);
            auto *up_data = static_cast<float *>(wu->data);
            for (int64_t i = 0; i < ggml_nelements(wg); ++i) {
                std::swap(gate_data[i], up_data[i]);
            }
        }
    }

    void Build(const std::string &name) {
        Require(context != nullptr, "GGML context creation failed");
        if (name.rfind("ffn-", 0) == 0 || name.rfind("bad-ffn-", 0) == 0 ||
            name.rfind("bad-region-", 0) == 0) {
            FFN(name);
        } else if (name.rfind("silu", 0) == 0 || name.rfind("mul-", 0) == 0 ||
                   name.rfind("glu", 0) == 0 || name.rfind("bad-silu", 0) == 0 ||
                   name.rfind("bad-mul-", 0) == 0 || name.rfind("bad-glu", 0) == 0) {
            Pointwise(name);
        } else {
            Matmul(name);
        }
    }

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
        const bool region = std::string(argv[2]).find("region") != std::string::npos;
        const char *entry = region ? "coral_check_ggml_ffn_import" : "coral_check_ggml_import";
        auto check = reinterpret_cast<CoralCheckGGMLImport>(dlsym(library, entry));
        Require(check != nullptr, "compiler check library has no test entry point");
        Fixture fixture;
        fixture.Build(argv[2]);
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
