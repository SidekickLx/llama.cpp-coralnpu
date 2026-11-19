// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "contracts/ffn_artifact.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>
#include <set>
#include <vector>

namespace coral::compiler {
namespace {
bool Diagnostic(llvm::StringRef name) {
    return name == "coral.ggml_tensor_id" || name == "coral.ggml_op" ||
           name == "coral.ggml_op_params";
}
mlir::DictionaryAttr SemanticAttributes(mlir::Operation *op) {
    llvm::SmallVector<mlir::NamedAttribute> attributes;
    for (auto attr : op->getAttrs()) {
        if (!Diagnostic(attr.getName().strref())) attributes.push_back(attr);
    }
    return mlir::DictionaryAttr::get(op->getContext(), attributes);
}
std::string AttributeText(mlir::Attribute attr) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    attr.print(stream);
    return text;
}

// Match the actual SSA graph against the standard typed FFN definition.
// This is a domain-specific contract check, not an IR or an evaluator.
// Properties (including dot dimensions/precision), attributes, types, exact
// argument roles, sharing and the entire operation set must agree.
struct Match {
    std::map<mlir::Operation *, mlir::Operation *> mapping;
    std::set<mlir::Operation *> used;
    std::vector<mlir::Operation *> ordered;
    std::vector<mlir::Operation *> reversed;

    bool Value(mlir::Value expected, mlir::Value actual) {
        if (expected.getType() != actual.getType()) return false;
        if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(expected)) {
            auto candidate = llvm::dyn_cast<mlir::BlockArgument>(actual);
            return candidate && candidate.getArgNumber() == argument.getArgNumber();
        }
        auto *left = expected.getDefiningOp();
        auto *right = actual.getDefiningOp();
        if (!left || !right) return false;
        auto found = mapping.find(left);
        if (found != mapping.end()) return found->second == right;
        if (left->getName() != right->getName() ||
            left->getNumResults() != 1 || right->getNumResults() != 1 ||
            left->getNumOperands() != right->getNumOperands() ||
            right->getNumRegions() != 0 || used.count(right) ||
            left->getPropertiesAsAttribute() != right->getPropertiesAsAttribute() ||
            SemanticAttributes(left) != SemanticAttributes(right)) return false;
        bool flip = false;
        const auto name = left->getName().getStringRef();
        if ((name == "stablehlo.multiply" || name == "stablehlo.add") &&
            left->getNumOperands() == 2) {
            auto *first = left->getOperand(0).getDefiningOp();
            auto *candidate = right->getOperand(0).getDefiningOp();
            auto *second = right->getOperand(1).getDefiningOp();
            flip = first && candidate && second && first->getName() != candidate->getName() &&
                   first->getName() == second->getName();
        }
        for (unsigned i = 0; i < left->getNumOperands(); ++i) {
            if (!Value(left->getOperand(i), right->getOperand(flip ? 1 - i : i))) return false;
        }
        mapping.emplace(left, right);
        used.insert(right);
        ordered.push_back(right);
        if (flip) reversed.push_back(right);
        return true;
    }
};
bool FunctionAttributes(mlir::func::FuncOp function) {
    for (auto attr : function->getAttrs()) {
        auto name = attr.getName().strref();
        if (name != "sym_name" && name != "function_type" && name != "sym_visibility" &&
            name != "arg_attrs" && name != "res_attrs" && name != "coral.ffn_abi" &&
            name != "coral.numeric_policy" && name != "coral.ggml_graph_instance" &&
            name != "coral.ggml_graph_revision") return false;
    }
    auto abi = function->getAttrOfType<mlir::StringAttr>("coral.ffn_abi");
    auto policy = function->getAttrOfType<mlir::StringAttr>("coral.numeric_policy");
    if (!abi || abi.getValue() != contracts::kFFNAbi ||
        !policy || policy.getValue() != "f32_no_fast_math_v1") return false;
    const char *roles[] = {"activation", "gate_weight", "up_weight", "down_weight"};
    for (unsigned i = 0; i < 4; ++i) {
        const auto attributes = function.getArgAttrDict(i);
        if (!attributes) continue;
        for (auto attr : attributes) {
            auto name = attr.getName().strref();
            if (name == "coral.ffn_role") {
                auto role = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
                if (!role || role.getValue() != roles[i]) return false;
            } else if (name != "coral.ggml_tensor_id") return false;
        }
    }
    for (auto attr : function.getResultAttrs(0)) {
        if (attr.getName().strref() != "coral.ggml_tensor_id") return false;
    }
    return true;
}
} // namespace

mlir::LogicalResult PrepareFFNAOT(mlir::ModuleOp module,
                                  const contracts::FFNCompileSpec &spec,
                                  llvm::StringRef ir_path, llvm::StringRef metadata_path) {
    const auto &s = spec.shape;
    constexpr int64_t limit = 2147483647 / sizeof(float);
    if (s.m <= 0 || s.d <= 0 || s.f <= 0 || s.m > limit || s.d > limit ||
        s.f > limit || s.m > limit / s.d || s.m > limit / s.f || s.d > limit / s.f)
        return module.emitError("invalid FFN tensor byte bounds");
    auto expected = BuildFFN(*module.getContext(), spec);
    if (!expected || module->getNumRegions() != 1 || !module->getAttrs().empty() ||
        std::distance(module.getBody()->begin(), module.getBody()->end()) != 1)
        return module.emitError("expected exactly one FFN function");
    auto function = llvm::dyn_cast<mlir::func::FuncOp>(module.getBody()->front());
    auto reference = expected->lookupSymbol<mlir::func::FuncOp>(contracts::kFFNEntryPoint);
    if (!function || function.getName() != contracts::kFFNEntryPoint ||
        function.getFunctionType() != reference.getFunctionType() ||
        !function.isPublic() || !llvm::hasSingleElement(function.getBody()) ||
        !FunctionAttributes(function))
        return module.emitError("FFN function contract mismatch");
    auto result = llvm::dyn_cast<mlir::func::ReturnOp>(function.getBody().front().getTerminator());
    auto golden = llvm::cast<mlir::func::ReturnOp>(reference.getBody().front().getTerminator());
    Match match;
    if (!result || result.getNumOperands() != 1 || !result->getAttrs().empty() ||
        !match.Value(golden.getOperand(0), result.getOperand(0)) ||
        match.used.size() + 1 != function.getBody().front().getOperations().size())
        return module.emitError("program does not implement the admitted positional FFN semantics");

    llvm::json::Array operations;
    for (auto *op : match.ordered) {
        llvm::json::Object record{{"index", static_cast<int64_t>(operations.size())},
                                  {"operation", op->getName().getStringRef()}};
        for (auto attr : op->getAttrs()) {
            if (Diagnostic(attr.getName().strref()))
                record[attr.getName().strref()] = AttributeText(attr.getValue());
        }
        operations.push_back(std::move(record));
    }
    // Strip only diagnostic metadata after proving semantics. Keep the actual
    // source operations and normalize independent projection order and the two
    // accepted commutative spellings; do not regenerate a program from shape.
    for (auto *op : match.reversed) {
        auto first = op->getOperand(0);
        op->setOperand(0, op->getOperand(1));
        op->setOperand(1, first);
    }
    for (auto *op : match.ordered) {
        llvm::SmallVector<mlir::StringAttr> remove;
        for (auto attr : op->getAttrs()) {
            if (Diagnostic(attr.getName().strref())) remove.push_back(attr.getName());
        }
        for (auto name : remove) op->removeAttr(name);
        op->setLoc(mlir::UnknownLoc::get(module.getContext()));
        op->moveBefore(result);
    }
    function->removeAttr("coral.ggml_graph_instance");
    function->removeAttr("coral.ggml_graph_revision");
    for (unsigned i = 0; i < 4; ++i) function.setArgAttrs(i, mlir::DictionaryAttr::get(module.getContext()));
    function.setResultAttrs(0, mlir::DictionaryAttr::get(module.getContext()));
    function->setLoc(mlir::UnknownLoc::get(module.getContext()));
    result->setLoc(mlir::UnknownLoc::get(module.getContext()));
    module->setLoc(mlir::UnknownLoc::get(module.getContext()));
    if (mlir::failed(VerifyModule(module))) return mlir::failure();
    std::error_code error;
    llvm::raw_fd_ostream metadata(metadata_path, error, llvm::sys::fs::CD_CreateNew,
                                 llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    if (error) return module.emitError("cannot create source metadata");
    metadata << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{
        {"schema", "coral.ffn.source-map.v1"}, {"operations", std::move(operations)}})) << '\n';
    metadata.close();
    if (metadata.has_error()) {
        metadata.clear_error();
        return module.emitError("cannot write source metadata");
    }
    return WriteModule(module, ir_path);
}

mlir::LogicalResult BindFFNReflection(mlir::ModuleOp module, llvm::StringRef reflection,
                                      llvm::StringRef output) {
    auto function = module.lookupSymbol<mlir::func::FuncOp>(contracts::kFFNEntryPoint);
    if (!function || reflection.empty() || reflection.size() > 4096)
        return module.emitError("invalid FFN reflection request");
    mlir::OpBuilder builder(module.getContext());
    function->setAttr("iree.reflection", builder.getDictionaryAttr({
        builder.getNamedAttr(contracts::kFFNReflectionKey, builder.getStringAttr(reflection))}));
    return WriteModule(module, output);
}
} // namespace coral::compiler
