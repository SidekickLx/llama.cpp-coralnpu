# 贡献指南

本项目处于早期开发阶段，M1/M2 功能验收已完成。提出实现前，请先阅读 `docs/project-plan.md`，说明目标阶段、输入输出契约及验证方式。

- 提交原创源码、测试、文档和可复现的依赖变更。
- 不提交本机绝对路径、用户名、凭据、私人页面链接、下载的上游仓库、模型、构建缓存或编译输出。
- 本机配置保存在 `.gitignore` 已排除的位置；共享示例使用相对路径或环境变量。
- 修改依赖时同步更新 `dependencies.lock.json`，保留上游许可证及必要说明。
- 数值验证同时给出 CPU 参考、容差、实际 NPU placement 和执行证据；模拟耗时不作为硬件性能结论。
- PR 说明具体问题、最终行为、验证结果及尚未覆盖的范围。

提交前检查 `git diff --cached --check` 和 `git diff --cached`，确认待提交内容中没有本机信息或生成文件。GitHub 仓库创建后，维护者应启用 secret scanning 和 push protection。

## Local build artifacts

Keep local build outputs and dependency caches for incremental builds. Do not
remove them unless explicitly requested. They remain excluded from Git; source
cleanup for publication must not delete the local development environment.

When a local env.sh exists, source it before building so the same tool paths
and Bazel cache configuration are reused across terminal sessions.
