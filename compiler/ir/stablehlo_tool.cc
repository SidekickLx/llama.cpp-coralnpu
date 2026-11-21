// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "llvm/Support/raw_ostream.h"

namespace coral::compiler { int ImportGGMLSnapshot(const char *, const char *); }

int main(int argc, char **argv) {
    if (argc == 4 && llvm::StringRef(argv[1]) == "import-ggml-ffn") {
        return coral::compiler::ImportGGMLSnapshot(argv[2], argv[3]);
    }
    mlir::MLIRContext context;
    if (argc == 11 && llvm::StringRef(argv[1]) == "bucket-ffn") {
        coral::contracts::FFNCompileSpec spec;
        int64_t capacity = 0;
        if (llvm::StringRef(argv[3]).getAsInteger(10, spec.shape.m) ||
            llvm::StringRef(argv[4]).getAsInteger(10, spec.shape.d) ||
            llvm::StringRef(argv[5]).getAsInteger(10, spec.shape.f) ||
            llvm::StringRef(argv[6]).getAsInteger(10, capacity)) return 1;
        const llvm::StringRef layout(argv[7]);
        if (layout == "transposed")
            spec.weight_layout = coral::contracts::FFNWeightLayout::Transposed;
        else if (layout != "native") return 1;
        auto module = coral::compiler::ReadModule(context, argv[2]);
        return !module || mlir::failed(coral::compiler::SpecializeFFNRows(
            *module, spec, capacity, argv[8], argv[9], argv[10])) ? 1 : 0;
    }
    if ((argc == 8 || argc == 9) && (llvm::StringRef(argv[1]) == "prepare-ffn" ||
                                    llvm::StringRef(argv[1]) == "optimize-ffn")) {
        coral::contracts::FFNCompileSpec spec;
        if (llvm::StringRef(argv[3]).getAsInteger(10, spec.shape.m) ||
            llvm::StringRef(argv[4]).getAsInteger(10, spec.shape.d) ||
            llvm::StringRef(argv[5]).getAsInteger(10, spec.shape.f)) return 1;
        if (argc == 9) {
            const llvm::StringRef layout(argv[8]);
            if (layout == "transposed")
                spec.weight_layout = coral::contracts::FFNWeightLayout::Transposed;
            else if (layout != "native") return 1;
        }
        auto module = coral::compiler::ReadModule(context, argv[2]);
        if (llvm::StringRef(argv[1]) == "optimize-ffn")
            return !module || mlir::failed(coral::compiler::OptimizeFFN(
                *module, spec, argv[6], argv[7])) ? 1 : 0;
        return !module || mlir::failed(coral::compiler::PrepareFFNAOT(
            *module, spec, argv[6], argv[7])) ? 1 : 0;
    }
    if (argc == 5 && llvm::StringRef(argv[1]) == "bind-ffn-reflection") {
        auto module = coral::compiler::ReadModule(context, argv[2]);
        return !module || mlir::failed(coral::compiler::BindFFNReflection(
            *module, argv[3], argv[4])) ? 1 : 0;
    }
    if (argc == 3 && llvm::StringRef(argv[1]) == "verify") {
        auto module = coral::compiler::ReadModule(context, argv[2]);
        return module ? 0 : 1;
    }
    if (argc == 6 && llvm::StringRef(argv[1]) == "emit-pointwise") {
        coral::contracts::PointwiseSpec spec;
        const llvm::StringRef kind(argv[2]);
        if (kind == "silu") {
            spec.kind = coral::contracts::PointwiseKind::Silu;
        } else if (kind == "multiply") {
            spec.kind = coral::contracts::PointwiseKind::Multiply;
        } else if (kind == "swiglu") {
            spec.kind = coral::contracts::PointwiseKind::SwiGLU;
        } else {
            llvm::errs() << "unknown pointwise operation\n";
            return 1;
        }
        if (llvm::StringRef(argv[3]).getAsInteger(10, spec.rows) ||
            llvm::StringRef(argv[4]).getAsInteger(10, spec.cols)) {
            llvm::errs() << "dimensions must be unsigned decimal integers\n";
            return 1;
        }
        auto module = coral::compiler::BuildPointwise(context, spec);
        return !module || mlir::failed(coral::compiler::WriteModule(*module, argv[5])) ? 1 : 0;
    }
    if (argc == 6) {
        int64_t a = 0, b = 0, c = 0;
        if (llvm::StringRef(argv[2]).getAsInteger(10, a) ||
            llvm::StringRef(argv[3]).getAsInteger(10, b) ||
            llvm::StringRef(argv[4]).getAsInteger(10, c)) {
            llvm::errs() << "dimensions must be decimal integers\n";
            return 1;
        }
        mlir::OwningOpRef<mlir::ModuleOp> module;
        const llvm::StringRef command(argv[1]);
        if (command == "emit-ffn") {
            coral::contracts::FFNCompileSpec spec;
            spec.shape = {a, b, c};
            module = coral::compiler::BuildFFN(context, spec);
        } else if (command == "emit-matmul") {
            module = coral::compiler::BuildMatmul(context, a, b, c);
        } else {
            llvm::errs() << "unknown command\n";
            return 1;
        }
        if (!module) {
            return 1;
        }
        return mlir::failed(coral::compiler::WriteModule(*module, argv[5])) ? 1 : 0;
    }
    llvm::errs() << "       coral-stablehlo-tool prepare-ffn INPUT M D F NEW_IR NEW_METADATA\n"
                 << "       coral-stablehlo-tool bind-ffn-reflection INPUT REFLECTION NEW_IR\n";
    llvm::errs() << "Usage: coral-stablehlo-tool emit-ffn M D F NEW_FILE\n"
                 << "       coral-stablehlo-tool emit-matmul M K N NEW_FILE\n"
                 << "       coral-stablehlo-tool emit-pointwise silu|multiply|swiglu M N NEW_FILE\n"
                 << "       coral-stablehlo-tool verify FILE\n";
    return 1;
}
