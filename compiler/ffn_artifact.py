# SPDX-License-Identifier: MIT
"""Wire-format assembly; the C++ loader is the authoritative runtime validator."""
import hashlib
import json

FILES = ("region.vmfb", "region.stablehlo.mlir", "region.metadata.json", "placement.json")
SCHEMA = "coral.ffn.artifact.v1"
PIPELINE = "coral.ffn.aot.f32.v1"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def reflection(description, semantic):
    return (description["binding_tag"] + ";policy=f32-no-fast-math-v1;target=" +
            description["compatibility_sha256"] + ";program=" + semantic + ";pipeline=" + PIPELINE)

def make_manifest(directory, description, compiler_digest, support_hashes, placement):
    hashes = {name: digest((directory / name).read_bytes()) for name in FILES}
    return {
        "schema": SCHEMA, "state": "complete", "contract": description["contract"],
        "compatibility": description["compatibility"],
        "program": {"pipeline": PIPELINE, "semantic_sha256": hashes["region.stablehlo.mlir"]},
        "compiler": {"binary_sha256": compiler_digest, "support_sha256": support_hashes,
                     "options": description["compiler_options"]},
        "files": hashes, "placement": placement,
        "reflection": reflection(description, hashes["region.stablehlo.mlir"]),
    }

def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
