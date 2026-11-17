// SPDX-License-Identifier: MIT
#pragma once
#include "ggml.h"
#include <stddef.h>
#include <stdint.h>
// Test-only C boundary. Borrowed tensors and CPU data outlive this call.
struct CoralGGMLFixture {
    uint32_t version;
    const char *name;
    const struct ggml_tensor *const *inputs;
    size_t input_count;
    const struct ggml_tensor *const *operations;
    size_t operation_count;
    const struct ggml_tensor *const *outputs;
    size_t output_count;
};
typedef int (*CoralCheckGGMLImport)(const struct CoralGGMLFixture *, const char *);
