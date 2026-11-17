// SPDX-License-Identifier: MIT
#include "graph_export.h"
#include "support/json.h"
#include "llama-context.h"
#include <exception>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace coral::ggml_backend {
namespace {
using coral::JsonString;
using Tensor = const ggml_tensor *;
struct Snapshot {
    std::vector<Tensor> tensors;
    std::set<Tensor> seen;
    void Visit(Tensor tensor) {
        if (!tensor || !seen.insert(tensor).second) {
            return;
        }
        tensors.push_back(tensor);
        for (auto *src : tensor->src) {
            Visit(src);
        }
        Visit(tensor->view_src);
    }
};
} // namespace

struct GraphExporter::Impl {
    std::filesystem::path output;
    ggml_backend_sched_t scheduler = nullptr;
    std::map<Tensor, std::size_t> ids;
    std::map<ggml_backend_buffer_t, std::size_t> buffers;
    std::exception_ptr error;
    std::string source;
    std::vector<std::string> splits;
    std::size_t generation = 0;
    int step = -1;
    bool rebuilt = false;

    explicit Impl(const std::filesystem::path &path) : output(path) {}

    std::size_t Id(Tensor tensor) {
        if (!tensor) {
            return 0;
        }
        const auto found = ids.find(tensor);
        if (found != ids.end()) {
            return found->second;
        }
        const auto id = ids.size() + 1;
        ids.emplace(tensor, id);
        return id;
    }

    void WriteTensor(std::ostream &out, Tensor tensor, const Snapshot &snapshot) {
        // Custom parameters may contain function/user pointers. Fail closed
        // instead of serializing opaque process addresses into a public dump.
        if (tensor->op == GGML_OP_MAP_CUSTOM1 || tensor->op == GGML_OP_MAP_CUSTOM2 ||
            tensor->op == GGML_OP_MAP_CUSTOM3 || tensor->op == GGML_OP_CUSTOM) {
            throw std::runtime_error("graph export does not support opaque custom op parameters");
        }
        out << "{\"id\":" << Id(tensor) << ",\"name\":" << JsonString(tensor->name)
            << ",\"op\":" << JsonString(ggml_op_name(tensor->op))
            << ",\"dtype\":" << JsonString(ggml_type_name(tensor->type))
            << ",\"flags\":" << tensor->flags
            << ",\"contiguous\":" << (ggml_is_contiguous(tensor) ? "true" : "false")
            << ",\"ne\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            out << (d ? "," : "") << tensor->ne[d];
        }
        out << "],\"nb\":[";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            out << (d ? "," : "") << tensor->nb[d];
        }
        out << "],\"op_params\":[";
        for (std::size_t i = 0; i < sizeof(tensor->op_params) / sizeof(int32_t); ++i) {
            out << (i ? "," : "") << tensor->op_params[i];
        }
        out << "],\"src\":[";
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            out << (i ? "," : "") << Id(tensor->src[i]);
        }
        out << "],\"view_src\":" << Id(tensor->view_src)
            << ",\"view_offset\":" << tensor->view_offs << ",\"consumers\":[";
        bool first = true;
        for (auto *consumer : snapshot.tensors) {
            for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
                if (consumer->src[slot] == tensor) {
                    out << (first ? "" : ",") << "{\"id\":" << Id(consumer)
                        << ",\"slot\":" << slot << '}';
                    first = false;
                }
            }
        }
        out << "],\"buffer\":";
        auto *buffer = tensor->buffer;
        if (!buffer && tensor->view_src) {
            buffer = tensor->view_src->buffer;
        }
        if (!buffer) {
            out << "null";
        } else {
            auto found = buffers.find(buffer);
            if (found == buffers.end()) {
                found = buffers.emplace(buffer, buffers.size() + 1).first;
            }
            out << "{\"id\":" << found->second
                << ",\"bytes\":" << ggml_backend_buffer_get_size(buffer)
                << ",\"type\":" << JsonString(ggml_backend_buffer_name(buffer)) << '}';
        }
        out << '}';
    }

    std::string Capture(ggml_cgraph *graph) {
        Snapshot snapshot;
        // Assign node IDs first. Dependency traversal must not depend on
        // allocation addresses or the ordering of a pointer-keyed map.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            Id(ggml_graph_node(graph, i));
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            snapshot.Visit(ggml_graph_node(graph, i));
        }
        for (auto *tensor : snapshot.tensors) {
            Id(tensor);
        }
        std::ostringstream out;
        out << "{\"nodes\":[";
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            out << (i ? "," : "") << Id(ggml_graph_node(graph, i));
        }
        out << "],\"tensors\":[";
        bool first = true;
        for (auto *tensor : snapshot.tensors) {
            out << (first ? "" : ",");
            WriteTensor(out, tensor, snapshot);
            first = false;
        }
        out << "]}";
        return out.str();
    }

    static void Observe(ggml_backend_sched_t scheduler, ggml_cgraph *graph,
                        int split_index, ggml_backend_t backend, void *user_data) noexcept {
        auto &self = *static_cast<Impl *>(user_data);
        if (self.error) {
            return;
        }
        try {
            if (self.step < 0 || scheduler != self.scheduler) {
                throw std::runtime_error("graph observer invoked outside an active step");
            }
            if (split_index < 0) {
                if (!self.splits.empty()) {
                    throw std::runtime_error("multiple executed graphs in one export step are unsupported");
                }
                self.ids.clear();
                self.buffers.clear();
                self.source = self.Capture(graph);
                ++self.generation;
                self.rebuilt = true;
            } else {
                if (self.source.empty() || split_index != static_cast<int>(self.splits.size())) {
                    throw std::runtime_error("missing original graph or out-of-order scheduler split");
                }
                std::ostringstream out;
                out << "{\"id\":" << split_index << ",\"backend\":"
                    << JsonString(ggml_backend_name(backend)) << ",\"graph\":"
                    << self.Capture(graph) << '}';
                self.splits.push_back(out.str());
            }
        } catch (...) {
            self.error = std::current_exception();
        }
    }
};

GraphExporter::GraphExporter(const std::filesystem::path &output)
    : impl_(std::make_unique<Impl>(output)) {}

GraphExporter::~GraphExporter() {
    if (impl_->scheduler) {
        ggml_backend_sched_set_graph_observer(impl_->scheduler, nullptr, nullptr);
    }
}

void GraphExporter::Attach(llama_context *context) {
    Attach(context->get_sched());
}

void GraphExporter::Attach(ggml_backend_sched_t scheduler) {
    if (!scheduler || impl_->scheduler) {
        throw std::invalid_argument("graph exporter requires one live scheduler");
    }
    impl_->scheduler = scheduler;
    ggml_backend_sched_set_graph_observer(scheduler, Impl::Observe, impl_.get());
}

void GraphExporter::Begin(int step) {
    if (!impl_->scheduler || step < 0 || impl_->step >= 0) {
        throw std::invalid_argument("invalid graph export step lifecycle");
    }
    impl_->step = step;
    impl_->splits.clear();
    impl_->rebuilt = false;
}

void GraphExporter::Finish() {
    if (impl_->error) {
        std::rethrow_exception(impl_->error);
    }
    if (impl_->step < 0 || impl_->splits.empty() ||
        impl_->splits.size() != static_cast<std::size_t>(ggml_backend_sched_get_n_splits(impl_->scheduler))) {
        throw std::runtime_error("incomplete graph/schedule observation");
    }
    const auto path = impl_->output / ("graph-" + std::to_string(impl_->step) + ".json");
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("refusing to overwrite a graph export");
    }
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path);
    out << "{\"schema\":\"coral-ggml-graph-v1\",\"step\":" << impl_->step
        << ",\"phase\":" << JsonString(impl_->step == 0 ? "prefill" : "decode")
        << ",\"graph_generation\":" << impl_->generation
        << ",\"graph_rebuilt\":" << (impl_->rebuilt ? "true" : "false")
        << ",\"source_stage\":\"before_scheduler_split\",\"source\":" << impl_->source
        << ",\"schedule_stage\":\"before_split_execution\",\"splits\":[";
    for (std::size_t i = 0; i < impl_->splits.size(); ++i) {
        out << (i ? "," : "") << impl_->splits[i];
    }
    out << "]}\n";
    out.close();
    impl_->step = -1;
}
} // namespace coral::ggml_backend
