// SPDX-License-Identifier: MIT
#include "ggml_importer.h"
#include "stablehlo_program.h"
#include "support/strict_json.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <set>
#include <vector>

namespace coral::compiler {
int ImportGGMLSnapshot(const char *input, const char *output) {
    using namespace coral::json;
    try {
        const auto wire = Read(input, 64 * 1024);
        Keys(wire, {"schema", "shape", "inputs", "nodes", "operations", "output"});
        Require(wire.at("schema") == "coral.ggml.ffn-snapshot.v1", "unknown snapshot schema");
        const auto &rows = wire.at("nodes");
        Require(rows.is_array() && rows.size() >= 5 && rows.size() <= 9, "invalid snapshot node count");
        std::vector<ggml_tensor> tensors(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto &row = rows[i];
            Keys(row, {"type", "ne", "nb", "op", "op_params", "src"});
            auto &tensor = tensors[i];
            tensor.type = static_cast<ggml_type>(Integer(row.at("type"), GGML_TYPE_F32, GGML_TYPE_F32));
            tensor.op = static_cast<ggml_op>(Integer(row.at("op"), 0, GGML_OP_COUNT - 1));
            for (const auto *name : {"ne", "nb"}) {
                Require(row.at(name).is_array() && row.at(name).size() == GGML_MAX_DIMS,
                        "invalid snapshot dimensions/strides");
            }
            for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
                tensor.ne[axis] = Integer(row.at("ne")[axis], 1, 2147483647);
                tensor.nb[axis] = Integer(row.at("nb")[axis], 1, 2147483647);
            }
            Require(row.at("op_params").is_array() &&
                        row.at("op_params").size() == GGML_MAX_OP_PARAMS / sizeof(int32_t),
                    "invalid snapshot parameters");
            for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++word) {
                tensor.op_params[word] = Integer(row.at("op_params")[word], INT32_MIN, INT32_MAX);
            }
            Require(row.at("src").is_array() && row.at("src").size() == GGML_MAX_SRC,
                    "invalid snapshot sources");
            for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
                // Only earlier nodes are legal; no cycles or future pointers.
                const auto id = Integer(row.at("src")[slot], 0, i);
                tensor.src[slot] = id ? &tensors[id - 1] : nullptr;
            }
        }
        GGMLFFNImportRequest request;
        request.graph = {1, 1}; // Transport-local IDs, removed by canonicalization.
        const auto &shape = wire.at("shape");
        Require(shape.is_array() && shape.size() == 3, "invalid snapshot profile");
        request.spec.shape = {Integer(shape[0], 1, 2147483647),
                              Integer(shape[1], 1, 2147483647),
                              Integer(shape[2], 1, 2147483647)};
        const auto &inputs = wire.at("inputs");
        Require(inputs.is_array() && inputs.size() == 4, "snapshot needs four roles");
        std::set<std::size_t> used;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto id = Integer(inputs[i], 1, tensors.size());
            Require(tensors[id - 1].op == GGML_OP_NONE, "boundary must be an argument");
            for (auto *source : tensors[id - 1].src) Require(!source, "argument has a producer");
            used.insert(id);
            request.inputs[i] = {&tensors[id - 1], {static_cast<std::uint64_t>(id)}};
        }
        const auto &operations = wire.at("operations");
        Require(operations.is_array() && (operations.size() == 4 || operations.size() == 5),
                "invalid FFN operation count");
        std::vector<GGMLTensorRef> refs;
        for (const auto &value : operations) {
            const auto id = Integer(value, 1, tensors.size());
            Require(used.insert(id).second, "repeated snapshot member");
            refs.push_back({&tensors[id - 1], {static_cast<std::uint64_t>(id)}});
        }
        Require(used.size() == tensors.size(), "unreferenced snapshot nodes");
        const auto output_id = Integer(wire.at("output"), 1, tensors.size());
        request.output = {&tensors[output_id - 1], {static_cast<std::uint64_t>(output_id)}};
        request.operations = refs;
        mlir::MLIRContext context;
        auto module = ImportGGMLFFNFunction(context, request);
        if (!module || mlir::failed(WriteModule(*module, output))) return 1;
        return 0;
    } catch (const std::exception &error) {
        llvm::errs() << "GGML snapshot: " << error.what() << '\n';
        return 1;
    }
}
} // namespace coral::compiler
