"""Generate a bounded dense-GEMM AOT catalogue; runs only during CMake configure.

General entries use runtime dimensions and core count. Bounded specializations
carry explicit dispatch guards. Layout, arithmetic and pipeline geometry are
template parameters. The same records generate device code and host dispatch.
"""

import argparse
import hashlib
import itertools
import json

from pathlib import Path


def configs(kind, major_a, major_b, epilogue, accumulate=False):
    bits = (16, 8, 8, 4)[kind]
    # Small/medium/large M tiles; dequant MN-major B needs a 256-element row.
    tiles = (
        [(64, 256), (128, 256)]
        if kind == 2 and major_b
        else ([(64, 64), (64, 128), (128, 128)] if bits == 4 else [(32, 64), (64, 128), (128, 128)])
    )
    bk = 128 if kind == 0 else 256
    shapes = [(bm, bn, bk, 1) for bm, bn in tiles]
    # Broader streaming pipelines for the common non-accumulating identity path.
    # These remain shape-independent and also support the FP32 output path.
    if epilogue == 0 and not accumulate:
        shapes += [(bm, bn, bk * 2, 2) for bm, bn in tiles]
        shapes += [(bm, bn, bk * 2, 2) for bm, bn in [(32, 128), (64, 64), (128, 256), (256, 128), (256, 256)]]
        if kind in (0, 1, 3):
            shapes += [(bm, bn, bk * 4, 1) for bm, bn in tiles]
        # Non-power-of-two tiles can fill the 28-Cube 950PR in one wave.
        shapes += (
            [(64, 80, 256, 1), (160, 256, 128, 1), (256, 160, 128, 1)]
            if kind == 0
            else [(64, 80, 1024, 1), (160, 256, 512, 2), (256, 160, 512, 2)]
            if kind == 3
            else [(64, 80, 512, 2), (160, 256, 256, 4), (256, 160, 256, 4)]
        )
    for bm, bn, bk, scale_blocks in dict.fromkeys(shapes):
        if kind == 2 and (bk > 512 or (bn if major_b else bk) not in (256, 512)):
            continue
        am = 256 // bits if major_a else 16
        an = 64 if epilogue == 2 else (256 // bits if major_b else 16)
        ka = 16 if kind == 0 else 64
        for sm, sn in [(1, 1), (1, 2), (2, 1)]:
            if bm % (sm * am) or bn % (sn * an):
                continue
            mm, mn = bm // sm, bn // sn
            if mm * mn * 4 + (bn * 4 if epilogue == 2 else 0) > 128 * 1024:
                continue
            mk = min(64 * 1024 * 8 // (2 * mm * bits), 64 * 1024 * 8 // (2 * mn * bits), max(ka, bk // 2))
            mk = mk // ka * ka
            while mk and bk % mk:
                mk -= ka
            if mk < min(bk, 2048 // bits) // 2:
                continue
            break
        else:
            continue
        sfk, sfs = (0, 0) if kind == 0 else (scale_blocks, 2)
        sfbytes = (bm + bn) * (bk // 64) * 2 * sfk * sfs
        stages = min(8 if kind == 0 else 4, (512 * 1024 - sfbytes) // ((bm + bn) * bk * bits // 8))
        # Two output buffers; legal for the at-most-two MAD tiles generated above.
        epi = 2
        if stages < 2:
            continue
        assert epi * (1 if kind == 2 else 2) >= sm * sn
        assert epi * (mm * mn * 4 + (bn * 4 if epilogue == 2 else 0)) <= 256 * 1024
        c = dict(
            block_m=bm,
            block_n=bn,
            block_k=bk,
            mad_m=mm,
            mad_n=mn,
            mad_k=mk,
            num_l1_stages=stages,
            num_l0_stages=2,
            sf_k_blocks=sfk,
            num_l1_sf_stages=sfs,
            num_epilogue_stages=epi,
            direct_store=False,
        )
        yield c
        if epilogue == 0 and not accumulate and (kind != 3 or not (major_a or major_b)):
            yield dict(c, direct_store=True)
            if kind == 2:
                yield dict(c, direct_store=True, dual_aiv_dequant=True)


def config_tag(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def write_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def specialization_matches(c, m, n, k, cores):
    return (
        (not c.get("shape_n") or c["shape_n"] == n)
        and (not c.get("shape_k") or c["shape_k"] == k)
        and (not c.get("num_cores") or c["num_cores"] == cores)
        and (not (c.get("single_m_block") or c.get("resident_a")) or m <= c["block_m"])
        and (not c.get("resident_b") or n <= c["block_n"])
    )


def generate(out, tuning_path=None):
    out.mkdir(parents=True, exist_ok=True)
    # Wheel notices come from the complete license comments kept with the code.
    for name, header in (("DEEPGEMM", "kernel/bf16_gemm.h"), ("FMT", "fmt/format.h")):
        source = Path(__file__).parent / header
        notice = source.read_text().split("/*", 1)[1].split("*/", 1)[0].strip()
        write_changed(out / f"{name}_LICENSE.txt", notice + "\n")
    records, host, declarations, sources = [], [], [], []
    specialization_path = Path(__file__).with_name("specializations.json")
    specializations = json.loads(specialization_path.read_text())
    used_specializations = set()
    fields = [
        "block_m",
        "block_n",
        "block_k",
        "mad_m",
        "mad_n",
        "mad_k",
        "num_l1_stages",
        "num_l0_stages",
        "num_epilogue_stages",
        "num_l1_sf_stages",
        "sf_k_blocks",
    ]
    for kind, ma, mb, output, (epi, acc) in itertools.product(
        range(4), range(2), range(2), range(2), [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    ):
        key = kind * 40 + ma * 20 + mb * 10 + output * 5 + [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)].index((epi, acc))
        kernel = ("bf16_gemm", "fp8_gemm", "fp8_dequant_gemm", "fp8_gemm")[kind]
        code = [f"#include <kernel/{kernel}.h>", '#include "aot_kernel.h"', "using namespace deep_gemm;"]
        a_type = ("bfloat16_t", "float8_e4m3_t", "float8_e4m3_t", "float4_e2m1x2_t")[kind]
        b_type = "float4_e2m1x2_t" if kind == 2 else a_type
        out_type = "float" if output else "bfloat16_t"
        a_major, b_major = ("Major::K", "Major::MN")[ma], ("Major::K", "Major::MN")[mb]
        entries = []
        for base in configs(kind, ma, mb, epi, acc):
            base_tag = config_tag(base)
            entries.append((base, base_tag))
            for i, spec in enumerate(specializations):
                if spec["key"] == key and spec["base_tag"] == base_tag:
                    allowed = {
                        "shape_n",
                        "shape_k",
                        "num_cores",
                        "single_m_block",
                        "resident_a",
                        "resident_b",
                        "sf_k_blocks",
                        "num_epilogue_stages",
                        "l2_a",
                        "l2_b",
                    }
                    if not spec["overrides"] or set(spec["overrides"]) - allowed:
                        raise ValueError(f"Invalid specialization: {spec}")
                    c = dict(base, **spec["overrides"])
                    if (c.get("resident_a") or c.get("resident_b")) and (kind != 0 or not c.get("shape_k")):
                        raise ValueError("BF16 residency needs a fixed K")
                    if c.get("single_m_block") and (kind == 0 or (kind == 2 and not c.get("dual_aiv_dequant"))):
                        raise ValueError("Unsupported single-M specialization")
                    for field in ("shape_n", "shape_k", "num_cores", "sf_k_blocks", "num_epilogue_stages"):
                        if field in spec["overrides"] and (type(c[field]) is not int or c[field] <= 0):
                            raise ValueError(f"Invalid positive specialization field: {field}")
                    if any(
                        c.get(field, "NORMAL_FIRST_VICTIM") not in ("NORMAL_FIRST_VICTIM", "NOTALLOC_KEEP")
                        for field in ("l2_a", "l2_b")
                    ):
                        raise ValueError("Unsupported cache hint")
                    entries.append((c, base_tag))
                    used_specializations.add(i)
        tags = set()
        for c, base_tag in entries:
            tag = config_tag(c)
            if tag in tags:
                raise ValueError(f"Duplicate AOT entry: {key}/{tag}")
            tags.add(tag)
            name = f"mojo_gemm_aot_{key}_{tag}"
            base_tiles = (
                [(64, 256), (128, 256)]
                if kind == 2 and mb
                else ([(64, 64), (64, 128), (128, 128)] if kind == 3 else [(32, 64), (64, 128), (128, 128)])
            )
            fallback = (
                tag == base_tag
                and (c["block_m"], c["block_n"]) in base_tiles
                and not c["direct_store"]
                and c["block_k"] == (128 if kind == 0 else 256)
                and c["sf_k_blocks"] == (0 if kind == 0 else 1)
            )
            params = [a_major, b_major, 0, c.get("shape_n", 0), c.get("shape_k", 0)] + [c[f] for f in fields[:8]]
            if kind:
                params += [c["sf_k_blocks"], c["num_l1_sf_stages"]]
            params += [
                c["num_epilogue_stages"],
                c.get("num_cores", 0),
                "asc_load_l2_cache_mode::" + c.get("l2_a", "NORMAL_FIRST_VICTIM"),
                "asc_load_l2_cache_mode::" + c.get("l2_b", "NORMAL_FIRST_VICTIM"),
                "asc_store_l2_cache_mode::NOTALLOC_CLEAN",
                str(bool(acc)).lower(),
                "GemmType::Normal",
            ]
            if kind in (1, 3):
                params += [a_type]
            params += [out_type, "epilogue::operators::" + ("Identity", "ScaleByAlpha", "AddBias")[epi], 256]
            params += [str(c["direct_store"]).lower()]
            if kind == 2:
                params += [str(c.get("dual_aiv_dequant", False)).lower()]
            if kind == 0:
                params += [str(c.get("resident_a", False)).lower(), str(c.get("resident_b", False)).lower()]
            else:
                params += [str(c.get("single_m_block", False)).lower()]
            args = [f"gm_ptr<{a_type},{a_major}>{{a.sa,a.a,0}}", f"gm_ptr<{b_type},{b_major}>{{a.sb,a.b,0}}"]
            if kind:
                args += ["reinterpret_cast<__gm__ int16_t*>(a.sfa)", "reinterpret_cast<__gm__ int16_t*>(a.sfb)"]
            args += [f"gm_ptr<{out_type}>{{a.sd,a.d,0}}", "a.m", "a.n", "a.k", "nullptr", "0", "a.epilogue", "cores"]
            signature = f"void {name}(const mojo_gemm::Args& a, uint32_t cores, void* stream)"
            declarations.append(signature + ";")
            code += [
                signature + " {",
                f"  {kernel}_impl<{', '.join(map(str, params))}>",
                f"      <<<cores, nullptr, stream>>>({', '.join(args)});",
                "}",
            ]
            init = ",".join(f".{f}={c[f]}" for f in fields)
            init += ",.direct_store=" + str(c["direct_store"]).lower()
            init += ",.dual_aiv_dequant=" + str(c.get("dual_aiv_dequant", False)).lower()
            for field in ("shape_n", "shape_k", "num_cores"):
                init += f",.{field}={c.get(field, 0)}"
            for field in ("single_m_block", "resident_a", "resident_b"):
                init += f",.{field}=" + str(c.get(field, False)).lower()
            init += (
                ",.l2_ctrl_a=asc_load_l2_cache_mode::"
                + c.get("l2_a", "NORMAL_FIRST_VICTIM")
                + ",.l2_ctrl_b=asc_load_l2_cache_mode::"
                + c.get("l2_b", "NORMAL_FIRST_VICTIM")
                + ",.l2_ctrl_store_cd=asc_store_l2_cache_mode::NOTALLOC_CLEAN"
            )
            host.append(f'{{{key}, "{tag}", {str(fallback).lower()}, GemmConfig{{{init}}}, &{name}}},')
            records.append(
                dict(
                    id=name,
                    tag=tag,
                    base_tag=base_tag,
                    fallback=fallback,
                    key=key,
                    kind=kind,
                    major_a=ma,
                    major_b=mb,
                    output=output,
                    epilogue=epi,
                    accumulate=acc,
                    **c,
                )
            )
        source = out / f"gemm_{key}.asc"
        write_changed(source, "\n".join(code) + "\n")
        sources.append(source.name)
    if len(used_specializations) != len(specializations):
        raise ValueError("Specialization refers to a missing base entry")
    # Scale conversion: both storage dtypes, both contiguous axes, all supported granularities.
    sfcode = [
        "#include <kernel/transform_sf.h>",
        '#include "aot_kernel.h"',
        "using namespace deep_gemm;",
        "void mojo_gemm_transform_sf(const mojo_gemm::ScaleArgs& a, uint32_t cores, void* stream) {",
        "switch(a.key) {",
    ]
    for floating, major, loggran in itertools.product(range(2), range(2), range(8)):
        gran = 1 << loggran
        key = floating * 16 + major * 8 + loggran
        bm = min(512, 8192 // gran) if major else (128 if gran == 1 else 256) // gran
        bk = (32 if gran == 1 else max(32 // gran, 4) * (2 if floating else 1)) if major else 128
        mt = ("Major::K", "Major::MN")[major]
        it = "uint32_t" if floating else "uint16_t"
        sfcode += [
            f"case {key}: transform_sf<{str(bool(floating)).lower()},{mt},{bm},{bk},GemmType::Normal,0,256,{gran}>",
            f"<<<cores,nullptr,stream>>>(gm_ptr<{it},{mt}>{{a.stride,a.src,a.batch_stride}},"
            "reinterpret_cast<__gm__ uint16_t*>(a.dst),nullptr,a.src_m,a.src_k,a.dst_m,a.dst_k,0,1,cores); break;",
        ]
    sfcode += ["}", "}"]
    write_changed(out / "transform.asc", "\n".join(sfcode) + "\n")
    sources.append("transform.asc")
    offsets = [next((i for i, r in enumerate(records) if r["key"] >= key), len(records)) for key in range(161)]
    offset_table = "static constexpr size_t aot_key_offsets[] = {" + ",".join(map(str, offsets)) + "};\n"
    write_changed(
        out / "registry.inc",
        "\n".join(declarations)
        + "\nnamespace deep_gemm {\nstatic const AotEntry aot_entries[] = {\n"
        + "\n".join(host)
        + "\n};\n"
        + offset_table
        + "}\n",
    )
    write_changed(out / "catalog.json", json.dumps(records, indent=2) + "\n")
    write_changed(
        out / "sources.cmake",
        "set(GEMM_AOT_SOURCES\n"
        + "\n".join(f'"${{CMAKE_CURRENT_BINARY_DIR}}/generated/{s}"' for s in sources)
        + "\n)\n",
    )
    tuning_path = tuning_path or Path(__file__).with_name("tuning.json")
    tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else []
    indexes = {(r["key"], r["tag"]): i for i, r in enumerate(records)}
    seen, tuned = set(), []
    for row in tuning:
        if not isinstance(row["soc"], str) or not row["soc"] or len(row["shape"]) != 3:
            raise ValueError("Tuning entries need a nonempty SoC and three dimensions")
        lookup = (row["soc"], row["cores"], row["key"], *row["shape"])
        if lookup in seen:
            raise ValueError(f"Duplicate tuning entry: {lookup}")
        seen.add(lookup)
        index = indexes[(row["key"], row["tag"])]
        if any(type(v) is not int or v <= 0 for v in (row["cores"], *row["shape"])):
            raise ValueError(f"Invalid tuning dimensions: {lookup}")
        if not specialization_matches(records[index], *row["shape"], row["cores"]):
            raise ValueError(f"Tuning does not satisfy specialization guards: {lookup}")
        tuned.append("{" + json.dumps(row["soc"]) + "," + ",".join(map(str, lookup[1:])) + "," + str(index) + "},")
    write_changed(
        out / "tuning.inc",
        "static const AotTuning aot_tuning[] = {\n" + "\n".join(tuned) + "\n{nullptr,0,0,0,0,0,0}\n};\n",
    )
    print(f"AOT: {len(records)} GEMM + 32 scale-transform instances, {len(sources)} translation units")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
