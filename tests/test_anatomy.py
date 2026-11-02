"""What a model's header says, and what that predicts, against numbers worked out by hand."""

import json
import os
import struct
import unittest

import helpers
from llmperfscope import adapters, anatomy, diagnose, driver, gguf, kernels, report

MIB = 1024 * 1024


def moe_model(path):
    """2 layers, 8 experts of which 2 are used: every number below follows from this."""
    return helpers.write_gguf(path, {
        "general.architecture": "toy",
        "general.name": "toy moe",
        "toy.block_count": 2,
        "toy.embedding_length": 64,
        "toy.attention.head_count": 4,
        "toy.expert_count": 8,
        "toy.expert_used_count": 2,
        "toy.context_length": 4096,
        "tokenizer.ggml.tokens": ["t"] * 300,
    }, [
        ("token_embd.weight", [64, 300], 8, 40 * MIB),
        ("blk.0.attn_q.weight", [64, 64], 8, 10 * MIB),
        ("blk.1.attn_q.weight", [64, 64], 8, 10 * MIB),
        ("blk.0.ffn_gate_inp.weight", [64, 8], 0, 1 * MIB),
        ("blk.0.ffn_up_exps.weight", [64, 128, 8], 39, 80 * MIB),
        ("blk.1.ffn_up_exps.weight", [64, 128, 8], 39, 80 * MIB),
        ("blk.0.ffn_up_shexp.weight", [64, 128], 8, 4 * MIB),
        ("blk.0.attn_norm.weight", [64], 0, 1 * MIB),
        ("output.weight", [64, 300], 8, 20 * MIB),
    ])


class Reader(unittest.TestCase):
    def test_header_and_tensors(self):
        path = moe_model(os.path.join(helpers.temp_dir(), "toy.gguf"))
        model = gguf.read(path)
        self.assertEqual(model["metadata"]["toy.expert_count"], 8)
        self.assertEqual(len(model["tensors"]), 9)
        by_name = {t["name"]: t for t in model["tensors"]}
        self.assertEqual(by_name["blk.0.ffn_up_exps.weight"]["bytes"], 80 * MIB)
        self.assertEqual(by_name["blk.0.ffn_up_exps.weight"]["type"], "MXFP4")
        self.assertEqual(by_name["output.weight"]["bytes"], 20 * MIB)

    def test_a_long_array_is_kept_as_its_length(self):
        path = helpers.write_gguf(os.path.join(helpers.temp_dir(), "v.gguf"), {"tokenizer.ggml.tokens": ["word"] * 1000, "a": 1}, [])
        model = gguf.read(path)
        self.assertEqual(model["metadata"]["tokenizer.ggml.tokens"], {"array_length": 1000})
        self.assertEqual(model["metadata"]["a"], 1)

    def test_a_split_model_is_one_model(self):
        directory = helpers.temp_dir()
        first = os.path.join(directory, "m-00001-of-00002.gguf")
        helpers.write_gguf(first, {"general.architecture": "toy", "split.count": 2}, [("blk.0.attn_q.weight", [4, 4], 0, MIB)])
        helpers.write_gguf(os.path.join(directory, "m-00002-of-00002.gguf"), {"split.no": 1}, [("blk.1.attn_q.weight", [4, 4], 0, 2 * MIB)])
        model = gguf.read(first)
        self.assertEqual(sorted(t["bytes"] for t in model["tensors"]), [MIB, 2 * MIB])

    def test_not_a_model(self):
        path = os.path.join(helpers.temp_dir(), "x.gguf")
        with open(path, "wb") as f:
            f.write(b"nope")
        self.assertIn("error", anatomy.describe(path))
        self.assertIsNone(anatomy.describe(os.path.join(helpers.temp_dir(), "missing.gguf")))
        self.assertIsNone(anatomy.describe(None))


class Parts(unittest.TestCase):
    def setUp(self):
        self.model = anatomy.describe(moe_model(os.path.join(helpers.temp_dir(), "toy.gguf")))
        self.parts = {row["part"]: row for row in self.model["parts"]}

    def test_shape(self):
        self.assertEqual((self.model["architecture"], self.model["layers"], self.model["experts"], self.model["experts_used"]), ("toy", 2, 8, 2))
        self.assertEqual(self.model["vocab"], 300)

    def test_tensors_land_in_their_part(self):
        self.assertEqual(self.parts["moe_experts"]["mib"], 160.0)
        self.assertEqual(self.parts["moe_shared"]["mib"], 4.0)
        self.assertEqual(self.parts["moe_router"]["mib"], 1.0)
        self.assertEqual(self.parts["attention"]["mib"], 20.0)
        self.assertEqual(self.parts["token_embedding"]["mib"], 40.0)
        self.assertEqual(self.parts["output_head"]["mib"], 20.0)
        self.assertEqual(self.parts["norm"]["mib"], 1.0)

    def test_a_token_reads_the_experts_it_was_routed_to(self):
        # 2 of 8 experts of 160 MiB, and all of the rest but the embedding table it reads one row of
        self.assertEqual(self.parts["moe_experts"]["read_mib"], 40.0)
        self.assertEqual(self.parts["token_embedding"]["read_mib"], 0.0)
        self.assertEqual(self.model["weights_read_per_token_mib"], 40.0 + 4.0 + 1.0 + 20.0 + 20.0 + 1.0)

    def test_parameters_a_token_uses(self):
        self.assertEqual(self.parts["moe_experts"]["active_params"], 2 * (64 * 128 * 8) // 4)
        self.assertEqual(self.parts["attention"]["active_params"], 2 * 64 * 64)

    def test_the_part_names_of_real_models(self):
        for name, part in (("blk.3.ffn_gate_exps.weight", "moe_experts"), ("blk.3.ffn_down_shexp.weight", "moe_shared"), ("blk.3.hc_ffn_fn.weight", "mixing"),
                           ("blk.3.attn_q_b.weight", "attention"), ("blk.3.indexer.attn_k.weight", "attention"), ("per_layer_token_embd.weight", "lookup_tables"),
                           ("blk.3.ffn_gate.weight", "feed_forward"), ("blk.3.attn_q_a_norm.weight", "norm"), ("output_norm.weight", "norm"), ("v.blk.0.attn_q.weight", "vision_audio")):
            self.assertEqual(anatomy._part_of(name, anatomy.GGUF_PARTS), part, name)


class Expectations(unittest.TestCase):
    def run_with(self, **extra):
        model = anatomy.describe(moe_model(os.path.join(helpers.temp_dir(), "toy.gguf")))
        run = {"model": model, "devices": [{"index": 0, "memory_gb_per_s": 100.0}, {"index": 1, "memory_gb_per_s": 100.0}], "devices_used": [0, 1],
               "throughput": {"decode": {"tokens_per_s": 100.0}}}
        run.update(extra)
        return run

    def test_floor_is_bytes_over_bandwidth(self):
        expected = anatomy.expectations(self.run_with())
        floor = 86.0 * MIB / 100e9 * 1000.0
        self.assertAlmostEqual(expected["decode_floor_ms"], floor, places=3)
        self.assertEqual(expected["decode_ms"], 10.0)
        self.assertAlmostEqual(expected["decode_share_of_time_reading"], floor / 10.0, places=2)

    def test_tensor_split_shares_it_and_layer_split_does_not(self):
        layer = anatomy.expectations(self.run_with(engine_config={"split_mode": "layer"}))
        tensor = anatomy.expectations(self.run_with(engine_config={"split_mode": "tensor"}))
        self.assertEqual((layer["gpus_sharing"], tensor["gpus_sharing"]), (1, 2))
        self.assertAlmostEqual(tensor["decode_floor_ms"], layer["decode_floor_ms"] / 2, places=3)

    def test_a_figure_given_by_hand_wins_and_says_so(self):
        expected = anatomy.expectations(self.run_with(memory_gb_per_s_given=50.0))
        self.assertEqual(expected["memory_bandwidth"], {"gb_per_s": 50.0, "source": "given on the command line"})

    def test_no_bandwidth_no_floor(self):
        expected = anatomy.expectations(self.run_with(devices=[{"index": 0}], devices_used=[0]))
        self.assertNotIn("decode_floor_ms", expected)
        self.assertEqual(expected["weights_read_per_token_mib"], 86.0)

    def test_prefill_rate_of_parameters(self):
        expected = anatomy.expectations(self.run_with(throughput={"prefill": {"tokens_per_s": 1000.0}}))
        params = expected["active_params_per_token"]
        self.assertEqual(expected["prefill_tera_params_per_s"], round(params * 1000.0 / 1e12, 3))

    def test_findings(self):
        run = self.run_with()
        run["expected"] = anatomy.expectations(run)
        found = {f["what"]: f for f in diagnose.structure_findings(run)}
        self.assertIn("decode_against_memory_bound", found)
        # experts are 160 of 246 MiB, and 40 of the 86 MiB a token reads
        self.assertAlmostEqual(found["moe_share_of_a_token"]["evidence"]["read_share"], 40 / 86, places=2)

    def test_the_report_shows_the_model(self):
        run = self.run_with(status="ok")
        text = report.render(run)
        self.assertIn("## Model", text)
        self.assertIn("8 experts per layer, 2 used per token", text)
        self.assertIn("| moe_experts | 160 |", text)


class HuggingFace(unittest.TestCase):
    def directory(self):
        directory = helpers.temp_dir()
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({"model_type": "toy_moe", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
                       "num_experts": 4, "num_experts_per_tok": 1, "vocab_size": 100}, f)
        names = {"model.embed_tokens.weight": 8 * MIB, "model.layers.0.self_attn.q_proj.weight": 4 * MIB, "model.layers.0.mlp.gate.weight": MIB,
                 "model.layers.0.mlp.experts.0.up.weight": 2 * MIB, "model.layers.0.mlp.experts.1.up.weight": 2 * MIB,
                 "model.layers.0.mlp.experts.2.up.weight": 2 * MIB, "model.layers.0.mlp.experts.3.up.weight": 2 * MIB, "lm_head.weight": 8 * MIB}
        header = {}
        offset = 0
        for name, size in names.items():
            header[name] = {"dtype": "F16", "shape": [size // 2], "data_offsets": [offset, offset + size]}
            offset += size
        data = json.dumps(header).encode()
        with open(os.path.join(directory, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(data)) + data)
        return directory

    def test_config_and_tensors(self):
        model = anatomy.describe(self.directory())
        parts = {row["part"]: row for row in model["parts"]}
        self.assertEqual((model["architecture"], model["layers"], model["experts"], model["experts_used"]), ("toy_moe", 2, 4, 1))
        self.assertEqual(parts["moe_experts"]["mib"], 8.0)
        self.assertEqual(parts["moe_experts"]["read_mib"], 2.0)
        self.assertEqual(parts["moe_router"]["mib"], 1.0)
        self.assertEqual(parts["attention"]["mib"], 4.0)
        self.assertEqual(parts["output_head"]["mib"], 8.0)

    def test_found_by_name_in_the_local_cache(self):
        hub = helpers.temp_dir()
        snapshot = os.path.join(hub, "models--org--toy", "snapshots", "abc")
        os.makedirs(snapshot)
        with open(os.path.join(snapshot, "config.json"), "w") as f:
            json.dump({"model_type": "toy", "num_hidden_layers": 3}, f)
        model = anatomy.describe("org/toy", environ={"HF_HUB_CACHE": hub})
        self.assertEqual((model["architecture"], model["layers"]), ("toy", 3))
        self.assertIsNone(anatomy.describe("org/other", environ={"HF_HUB_CACHE": hub}))


class EngineSettings(unittest.TestCase):
    def test_llama_server_flags(self):
        cmd = ["taskset", "-c", "1", "llama-server", "-m", "m.gguf", "-ngl", "99", "-sm", "tensor", "-c", "8192", "-ub", "512", "-fa", "1", "-np", "2",
               "-cmoe", "-md", "d.gguf", "--spec-type", "draft-dspark", "--spec-draft-n-max", "4", "-ctxcp", "8"]
        config = adapters.select("auto", cmd).engine_config()
        self.assertEqual(config["split_mode"], "tensor")
        self.assertEqual(config["ctx_size"], "8192")
        self.assertEqual(config["ubatch_size"], "512")
        self.assertEqual(config["draft_model"], "d.gguf")
        self.assertEqual(config["spec_draft_n_max"], "4")
        self.assertEqual(config["ctx_checkpoints"], "8")
        self.assertTrue(config["cpu_moe"])

    def test_a_flag_given_twice_counts_as_the_last(self):
        config = adapters.select("auto", ["llama-server", "-c", "16384", "-m", "m.gguf", "-c", "8192"]).engine_config()
        self.assertEqual(config["ctx_size"], "8192")

    def test_vllm_tensor_parallel_counts_as_a_tensor_split(self):
        config = adapters.select("auto", ["vllm", "serve", "org/m", "--tensor-parallel-size", "4", "--max-model-len", "1000"]).engine_config()
        self.assertEqual((config["split_mode"], config["ctx_size"]), ("tensor", "1000"))

    def test_an_engine_it_does_not_know(self):
        self.assertEqual(adapters.select("auto", ["python", "run.py"]).engine_config(), {})

    def test_counters_from_the_log(self):
        adapter = adapters.select("auto", ["llama-server", "-m", "m.gguf"])
        adapter.on_line("stdout", 0.0, '{"type":"log","msg":"slot print_timing: id  0 | task 9 |    graphs reused =         31\\n"}', lambda *a, **k: None)
        adapter.on_line("stdout", 0.0, '{"type":"log","msg":"slot print_timing: id  0 | task 9 |   graph reallocs =          2    /     7        (same size) / (changed size)\\n"}', lambda *a, **k: None)
        adapter.on_line("stdout", 0.0, "slot create_check: created context checkpoint 1 of 32 (pos_min = 0)", lambda *a, **k: None)
        adapter.on_line("stdout", 0.0, "nothing to see", lambda *a, **k: None)
        self.assertEqual(adapter.report["engine_counters"], {"graphs_reused": 31, "graph_reallocs_same_size": 2, "graph_reallocs_changed": 7, "checkpoints_created": 1})


class Components(unittest.TestCase):
    def test_names_of_real_kernels(self):
        for name, component in (("flash_attn_ext_f16", "attention"), ("soft_max_f32", "attention"), ("mul_mat_vec_q", "matrix multiply, dense or expert"),
                                ("mul_mat_q", "matrix multiply, dense or expert"), ("quantize_mmq_q8_1", "matrix multiply, dense or expert"), ("topk_moe_weights", "expert routing and sums"),
                                ("argsort_f32_i32", "expert routing and sums"), ("dsv4_hc_post_f32", "mixing between streams"), ("rms_norm_f32", "normalization"),
                                ("get_rows_cuda_q", "table lookups"), ("set_rows_cuda", "KV cache writes"), ("scale_f32", "element-wise"), ("cpy_scalar", "copies and reshapes"),
                                ("ncclDevKernel_AllReduce_f32_RING", "collectives"), ("ggml_cuda_ar_ll_kernel", "collectives"), ("some_torch_kernel", "other")):
            self.assertEqual(kernels.component_of(name), component, name)

    def test_matrix_multiplies_against_the_memory(self):
        line = kernels._against_memory({"expected": {"decode_floor_ms": 2.0}}, "decode", {"matrix multiply, dense or expert": 4.0})
        self.assertIn("run at 50% of what the memory allows", line)
        self.assertIn("not all of them run", kernels._against_memory({"expected": {"decode_floor_ms": 5.0}}, "decode", {"matrix multiply, dense or expert": 4.0}))
        self.assertIsNone(kernels._against_memory({"expected": {"decode_floor_ms": 2.0}}, "prefill", {"matrix multiply, dense or expert": 4.0}))
        self.assertIsNone(kernels._against_memory({}, "decode", {"matrix multiply, dense or expert": 4.0}))

    def test_totals(self):
        rows = [{"name": "mul_mat_vec_q", "cost_ms_per_token": 3.0}, {"name": "mul_mat_q", "cost_ms_per_token": 1.0}, {"name": "rms_norm_f32", "cost_ms_per_token": 0.5},
                {"name": "ncclDevKernel_AllReduce", "cost_ms_per_token": 1.5}]
        self.assertEqual(kernels.by_component(rows), [("matrix multiply, dense or expert", 4.0), ("collectives", 1.5), ("normalization", 0.5)])


class Context(unittest.TestCase):
    @staticmethod
    def reply(prompt_tokens, gap_ms, n=20, spike_ms=None):
        gaps = [gap_ms / 1000.0] * n
        if spike_ms:
            gaps[0] = spike_ms / 1000.0
        return {"ok": True, "ttft_s": 0.1, "itl_s": gaps, "chunks": n + 1, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n + 1}}

    def test_the_cost_of_context(self):
        replies = [self.reply(n, 10.0 + 2.0 * n / 1000.0, spike_ms=500.0) for n in (100, 200, 400, 800, 1600, 3200, 6400, 12800)]
        line = driver.between_tokens_by_context(replies)["between_tokens_fit"]
        self.assertAlmostEqual(line["per_1000_context_tokens_ms"], 2.0, delta=0.05)
        self.assertAlmostEqual(line["base_ms"], 10.0, delta=0.05)

    def test_flat_with_a_window(self):
        replies = [self.reply(n, 10.0) for n in (100, 200, 400, 800, 1600, 3200, 6400, 12800)]
        self.assertAlmostEqual(driver.between_tokens_by_context(replies)["between_tokens_fit"]["per_1000_context_tokens_ms"], 0.0, places=3)

    def test_a_finding_when_it_grows(self):
        replies = [self.reply(n, 10.0 + 2.0 * n / 1000.0) for n in (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000)]
        summary = driver.summarize_load(replies, 1.0)
        summary["concurrency"] = 1
        found = {f["what"] for f in diagnose.structure_findings({"throughput": {"serve_c1": summary}})}
        self.assertIn("kv_cache_cost_grows_with_context", found)

    def test_speculative_tokens(self):
        replies = [dict(self.reply(100, 10.0), server_timings={"draft_n": 40, "draft_n_accepted": 30}), dict(self.reply(100, 10.0), server_timings={"draft_n": 10, "draft_n_accepted": 5})]
        self.assertEqual(driver.summarize_load(replies, 1.0)["speculative"], {"drafted": 50, "accepted": 35, "accepted_share": 0.7})

    def test_a_server_that_does_not_report_its_cache(self):
        replies = []
        for turn in range(2):
            item = self.reply(100 * (turn + 1), 10.0)
            item.update(turn=turn, ttft_s=0.02)
            replies.append(item)
        summary = driver.summarize_load(replies, 1.0)
        summary["concurrency"] = 1
        self.assertEqual([t["cached_share"] for t in summary["by_turn"]], [None, None])
        found = {f["what"]: f["text"] for f in diagnose.structure_findings({"throughput": {"serve_c1": summary}})}
        self.assertIn("does not report", found["first_token_by_turn"])
        self.assertNotIn("% cached", found["first_token_by_turn"])

    def test_a_conversation_asks_the_server_to_report_its_cache(self):
        self.assertEqual(adapters.SERVER_PRESETS["vllm"]["cache_report_args"], ["--enable-prompt-tokens-details"])
        self.assertEqual(adapters.SERVER_PRESETS["sglang"]["cache_report_args"], ["--enable-cache-report"])

    def test_the_floor_without_server_timers_uses_what_the_client_saw(self):
        model = {"weights_read_per_token_mib": 100.0, "parts": []}
        run = {"model": model, "devices": [{"index": 0, "memory_gb_per_s": 100.0}], "devices_used": [0],
               "throughput": {"serve_c1": {"concurrency": 1, "itl_ms": {"p50": 4.0}}}}
        self.assertEqual(anatomy.expectations(run)["decode_ms"], 4.0)
        run["throughput"]["serve_c1"]["speculative"] = {"drafted": 10, "accepted": 5, "accepted_share": 0.5}
        self.assertNotIn("decode_ms", anatomy.expectations(run))

    def test_cached_share_and_turns(self):
        replies = []
        for turn, (prompt, cached, ms) in enumerate(((100, 0, 200.0), (300, 96, 120.0), (500, 296, 130.0))):
            item = self.reply(prompt, 10.0)
            item.update(turn=turn, cached_tokens=cached, ttft_s=ms / 1000.0)
            replies.append(item)
        summary = driver.summarize_load(replies, 1.0)
        self.assertEqual(summary["prompt_cached_share"], round(392 / 900, 3))
        self.assertEqual([t["turn"] for t in summary["by_turn"]], [1, 2, 3])
        self.assertEqual(summary["by_turn"][1], {"turn": 2, "requests": 1, "prompt_tokens_mean": 300, "ttft_p50_ms": 120.0, "cached_share": 0.32})

    def test_speed_by_content(self):
        replies = []
        for language, gap in (("ko", 12.0), ("en", 10.0), ("rare", 50.0)):
            for _ in range(5 if language != "rare" else 2):
                replies.append(dict(self.reply(100, gap), language=language))
        summary = driver.summarize_load(replies, 1.0)
        summary["concurrency"] = 1
        rows = summary["by_content"]["language"]
        self.assertEqual([r["group"] for r in rows], ["en", "ko"])
        self.assertEqual(rows[0]["itl_p50_ms"], 10.0)
        self.assertEqual(rows[0]["first_token_ms_per_prompt_token"], 1.0)
        found = {f["what"]: f for f in diagnose.structure_findings({"throughput": {"serve_c1": summary}})}
        self.assertEqual(found["speed_depends_on_content"]["evidence"]["slowest"]["group"], "ko")

    def test_reserving_the_graph_every_request(self):
        run = {"engine_counters": {"graph_reallocs_changed": 30, "graph_reallocs_same_size": 2}, "throughput": {"serve_c1": {"concurrency": 1, "requests": 10}}}
        self.assertIn("graphs_reserved_per_request", {f["what"] for f in diagnose.structure_findings(run)})
        run["engine_counters"]["graph_reallocs_changed"] = 1
        self.assertNotIn("graphs_reserved_per_request", {f["what"] for f in diagnose.structure_findings(run)})


if __name__ == "__main__":
    unittest.main()
