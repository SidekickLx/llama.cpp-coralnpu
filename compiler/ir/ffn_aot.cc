// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "contracts/ffn_artifact.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"
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
bool FunctionAttributes(mlir::func::FuncOp function, const contracts::FFNCompileSpec &spec) {
    for (auto attr : function->getAttrs()) {
        auto name = attr.getName().strref();
        if (name != "sym_name" && name != "function_type" && name != "sym_visibility" &&
            name != "arg_attrs" && name != "res_attrs" && name != "coral.ffn_abi" &&
            name != "coral.numeric_policy" && name != "coral.ggml_graph_instance" &&
            name != "coral.ggml_graph_revision") return false;
    }
    auto abi = function->getAttrOfType<mlir::StringAttr>("coral.ffn_abi");
    auto policy = function->getAttrOfType<mlir::StringAttr>("coral.numeric_policy");
    if (!abi || abi.getValue() != contracts::FFNAbi(spec) ||
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
mlir::LogicalResult NormalizeFFN(mlir::ModuleOp module,
                                  const contracts::FFNCompileSpec &spec,
                                  llvm::json::Array &operations) {
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
        !FunctionAttributes(function, spec))
        return module.emitError("FFN function contract mismatch");
    auto result = llvm::dyn_cast<mlir::func::ReturnOp>(function.getBody().front().getTerminator());
    auto golden = llvm::cast<mlir::func::ReturnOp>(reference.getBody().front().getTerminator());
    Match match;
    if (!result || result.getNumOperands() != 1 || !result->getAttrs().empty() ||
        !match.Value(golden.getOperand(0), result.getOperand(0)) ||
        match.used.size() + 1 != function.getBody().front().getOperations().size())
        return module.emitError("program does not implement the admitted positional FFN semantics");

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
    return mlir::success();
}
mlir::LogicalResult WriteMetadata(mlir::ModuleOp module, llvm::StringRef path,
                                  llvm::json::Object value) {
    std::error_code error;
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::CD_CreateNew,
                               llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    if (error) return module.emitError("cannot create metadata");
    output << llvm::formatv("{0:2}", llvm::json::Value(std::move(value))) << '\n';
    output.close();
    if (output.has_error()) {
        output.clear_error();
        return module.emitError("cannot write metadata");
    }
    return mlir::success();
}
} // namespace

mlir::LogicalResult PrepareFFNAOT(mlir::ModuleOp module,
                                  const contracts::FFNCompileSpec &spec,
                                  llvm::StringRef ir_path, llvm::StringRef metadata_path) {
    llvm::json::Array operations;
    if (mlir::failed(NormalizeFFN(module, spec, operations))) return mlir::failure();
    if (mlir::failed(WriteMetadata(module, metadata_path, llvm::json::Object{
            {"schema", "coral.ffn.source-map.v1"}, {"operations", std::move(operations)}})))
        return mlir::failure();
    return WriteModule(module, ir_path);
}

mlir::LogicalResult OptimizeFFN(mlir::ModuleOp module,
                                const contracts::FFNCompileSpec &target,
                                llvm::StringRef ir_path, llvm::StringRef plan_path) {
    namespace hlo = mlir::stablehlo;
    // Only an identity F32 conversion is discarded. No narrowing/widening pair,
    // rounding change, reduction reassociation, or approximate math is allowed.
    // Verify the resulting *actual* SSA graph against the complete FFN contract
    // before changing its public weight bindings.
    mlir::IRRewriter rewriter(module.getContext());
    int64_t removed = 0;
    auto function = module.lookupSymbol<mlir::func::FuncOp>(contracts::kFFNEntryPoint);
    if (!function || !llvm::hasSingleElement(function.getBody()))
        return module.emitError("expected a single-block FFN");
    for (auto &op : llvm::make_early_inc_range(function.front())) {
        auto convert = llvm::dyn_cast<hlo::ConvertOp>(op);
        if (!convert || convert.getOperand().getType() != convert.getResult().getType())
            continue;
        const auto type = llvm::dyn_cast<mlir::RankedTensorType>(convert.getType());
        if (!type || !type.getElementType().isF32() || !SemanticAttributes(&op).empty())
            continue;
        rewriter.replaceOp(convert, convert.getOperand());
        ++removed;
    }
    contracts::FFNCompileSpec source = target;
    source.weight_layout = contracts::FFNWeightLayout::Native;
    llvm::json::Array source_ops;
    if (mlir::failed(NormalizeFFN(module, source, source_ops))) return mlir::failure();
    llvm::json::Array weights;
    if (target.weight_layout == contracts::FFNWeightLayout::Transposed) {
        llvm::SmallVector<mlir::Type> types(function.getArgumentTypes());
        for (unsigned i = 1; i < 4; ++i) {
            auto arg = function.getArgument(i);
            const auto old = llvm::cast<mlir::RankedTensorType>(arg.getType());
            const auto replacement = mlir::RankedTensorType::get(
                {old.getDimSize(1), old.getDimSize(0)}, old.getElementType());
            // Admission proved that each role has one RHS dot use. Check again
            // at the rewrite boundary rather than relying on an unchecked cast.
            if (!arg.hasOneUse()) return module.emitError("weight role has unexpected uses");
            auto dot = llvm::dyn_cast<hlo::DotGeneralOp>(*arg.getUsers().begin());
            if (!dot || dot.getRhs() != arg)
                return module.emitError("weight role is not a dot RHS");
            weights.push_back(llvm::json::Object{
                {"argument", i}, {"operation", "transpose"}, {"permutation", llvm::json::Array{1, 0}},
                {"source_shape", llvm::json::Array{old.getDimSize(0), old.getDimSize(1)}},
                {"prepared_shape", llvm::json::Array{old.getDimSize(1), old.getDimSize(0)}}});
            arg.setType(replacement);
            types[i] = replacement;
            dot.setDotDimensionNumbersAttr(hlo::DotDimensionNumbersAttr::get(
                module.getContext(), {}, {}, {1}, {0}));
        }
        function.setType(mlir::FunctionType::get(
            module.getContext(), types, function.getResultTypes()));
        function->setAttr("coral.ffn_abi",
                          rewriter.getStringAttr(contracts::FFNAbi(target)));
    }
    llvm::json::Array target_ops;
    if (mlir::failed(VerifyModule(module)) ||
        mlir::failed(NormalizeFFN(module, target, target_ops))) return mlir::failure();
    if (mlir::failed(WriteMetadata(module, plan_path, llvm::json::Object{
            {"schema", "coral.ffn.optimization-plan.v1"},
            {"pass", "P1-weight-layout"}, {"source_abi", contracts::kFFNAbi},
            {"target_abi", contracts::FFNAbi(target)},
            {"removed_identity_f32_converts", removed},
            {"weight_preparation", std::move(weights)},
            {"proof", "exact admitted FFN; transpose RHS storage and contract axis 0; no arithmetic change"}})))
        return mlir::failure();
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
