// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
    mlir::MLIRContext context;
    if (argc == 8 && llvm::StringRef(argv[1]) == "prepare-ffn") {
        coral::contracts::FFNCompileSpec spec;
        if (llvm::StringRef(argv[3]).getAsInteger(10, spec.shape.m) ||
            llvm::StringRef(argv[4]).getAsInteger(10, spec.shape.d) ||
            llvm::StringRef(argv[5]).getAsInteger(10, spec.shape.f)) return 1;
        auto module = coral::compiler::ReadModule(context, argv[2]);
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
