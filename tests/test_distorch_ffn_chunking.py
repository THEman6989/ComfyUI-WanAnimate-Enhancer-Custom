from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "wan_animate_to_video_enhanced.py"


class FakeModelPatcher:
    def __init__(self, model=None):
        self.object_patches = {}
        self.model = model

    def clone(self):
        clone = FakeModelPatcher(self.model)
        clone.object_patches = dict(self.object_patches)
        return clone

    def add_object_patch(self, path, value):
        self.object_patches[path] = value


def load_module():
    spec = importlib.util.spec_from_file_location("wan_enhancer_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model():
    ffn = torch.nn.Sequential(
        torch.nn.Linear(4, 9),
        torch.nn.GELU(),
        torch.nn.Linear(9, 3),
    )
    return SimpleNamespace(blocks=[SimpleNamespace(ffn=ffn)]), ffn


def test_metadata_detection_walks_model_layers_and_handles_cycles():
    module = load_module()
    root = SimpleNamespace()
    root.model = SimpleNamespace(diffusion_model=root)
    assert module.has_distorch_metadata(root) is False

    root.model._distorch_v2_meta = {"full_allocation": "x"}
    assert module.has_distorch_metadata(root) is True


def test_chunked_ffn_patch_is_numerically_equivalent_and_branch_local():
    module = load_module()
    torch.manual_seed(7)
    model, ffn = make_model()
    patcher = FakeModelPatcher()
    input_tensor = torch.randn(2, 11, 4)
    expected = ffn(input_tensor)
    original_function = ffn.forward.__func__

    enabled = module.configure_distorch_ffn_chunking(
        patcher, model, True, 3, "diffusion_model."
    )
    patched_forward = patcher.object_patches[
        "diffusion_model.blocks.0.ffn.forward"
    ]
    actual = patched_forward(input_tensor)

    assert enabled is True
    assert ffn.forward.__func__ is original_function
    assert len(patcher.object_patches) == 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_non_distorch_and_zero_size_register_no_patches():
    module = load_module()
    model, _ = make_model()

    normal_patcher = FakeModelPatcher()
    assert module.configure_distorch_ffn_chunking(
        normal_patcher, model, False, 4096
    ) is False
    assert normal_patcher.object_patches == {}

    disabled_patcher = FakeModelPatcher()
    assert module.configure_distorch_ffn_chunking(
        disabled_patcher, model, True, 0
    ) is False
    assert disabled_patcher.object_patches == {}


def test_sibling_patcher_is_not_modified_or_reset():
    module = load_module()
    model, ffn = make_model()
    distorch_branch = FakeModelPatcher()
    normal_branch = FakeModelPatcher()
    original_function = ffn.forward.__func__

    assert module.configure_distorch_ffn_chunking(
        distorch_branch, model, True, 2
    ) is True
    assert module.configure_distorch_ffn_chunking(
        normal_branch, model, False, 2
    ) is False

    assert len(distorch_branch.object_patches) == 1
    assert normal_branch.object_patches == {}
    assert ffn.forward.__func__ is original_function


def test_reconfiguration_creates_independent_branch_patches():
    module = load_module()
    model, ffn = make_model()
    input_tensor = torch.randn(1, 9, 4)
    expected = ffn(input_tensor)
    first = FakeModelPatcher()
    second = FakeModelPatcher()

    module.configure_distorch_ffn_chunking(first, model, True, 8)
    module.configure_distorch_ffn_chunking(second, model, True, 2)

    first_forward = first.object_patches["blocks.0.ffn.forward"]
    second_forward = second.object_patches["blocks.0.ffn.forward"]
    assert first_forward is not second_forward
    torch.testing.assert_close(first_forward(input_tensor), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(second_forward(input_tensor), expected, rtol=1e-5, atol=1e-6)


def test_autocast_output_dtype_matches_original_ffn():
    module = load_module()
    model, ffn = make_model()
    patcher = FakeModelPatcher()
    input_tensor = torch.randn(1, 11, 4)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = ffn(input_tensor)
        module.configure_distorch_ffn_chunking(patcher, model, True, 3)
        patched_forward = patcher.object_patches["blocks.0.ffn.forward"]
        actual = patched_forward(input_tensor)

    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_real_model_patcher_applies_and_restores_object_patch():
    module = load_module()
    from comfy.model_patcher import ModelPatcher

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(4, 9),
                torch.nn.GELU(),
                torch.nn.Linear(9, 3),
            )

    class Wan(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = Wan()

    root = Root()
    patcher = ModelPatcher(
        root, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    original_function = root.diffusion_model.blocks[0].ffn.forward.__func__
    input_tensor = torch.randn(1, 7, 4)
    expected = root.diffusion_model.blocks[0].ffn(input_tensor)

    assert module.configure_distorch_ffn_chunking(
        patcher, root.diffusion_model, True, 2, "diffusion_model."
    ) is True
    patcher.patch_model(load_weights=False)
    assert root.diffusion_model.blocks[0].ffn.forward.__func__ is not original_function
    torch.testing.assert_close(
        root.diffusion_model.blocks[0].ffn(input_tensor), expected, rtol=1e-5, atol=1e-6
    )
    patcher.unpatch_model(unpatch_weights=False)
    assert root.diffusion_model.blocks[0].ffn.forward.__func__ is original_function


def test_standalone_chunking_node_exposes_selectable_presets():
    module = load_module()
    inputs = module.WanAnimateFFNChunking.INPUT_TYPES()["required"]

    assert inputs["chunk_tokens"][0] == [
        "disabled",
        "512",
        "1024",
        "2048",
        "4096",
        "8192",
        "16384",
        "32768",
        "65536",
        "128000",
    ]
    assert module.WanAnimateFFNChunking.RETURN_TYPES == ("MODEL",)


def test_standalone_chunking_node_patches_normal_dynamic_branch():
    module = load_module()
    wan, ffn = make_model()
    root = SimpleNamespace(diffusion_model=wan)
    original = FakeModelPatcher(root)

    (chunked,) = module.WanAnimateFFNChunking().apply(original, "2048")

    assert chunked is not original
    assert original.object_patches == {}
    assert "diffusion_model.blocks.0.ffn.forward" in chunked.object_patches
    assert chunked._wan_ffn_chunk_tokens == 2048


def test_standalone_chunking_node_disabled_is_branch_local_passthrough():
    module = load_module()
    wan, _ = make_model()
    original = FakeModelPatcher(SimpleNamespace(diffusion_model=wan))

    (disabled,) = module.WanAnimateFFNChunking().apply(original, "disabled")

    assert disabled is not original
    assert disabled.object_patches == {}
    assert disabled._wan_ffn_chunk_tokens == 0


def main():
    tests = [
        test_metadata_detection_walks_model_layers_and_handles_cycles,
        test_chunked_ffn_patch_is_numerically_equivalent_and_branch_local,
        test_non_distorch_and_zero_size_register_no_patches,
        test_sibling_patcher_is_not_modified_or_reset,
        test_reconfiguration_creates_independent_branch_patches,
        test_autocast_output_dtype_matches_original_ffn,
        test_real_model_patcher_applies_and_restores_object_patch,
        test_standalone_chunking_node_exposes_selectable_presets,
        test_standalone_chunking_node_patches_normal_dynamic_branch,
        test_standalone_chunking_node_disabled_is_branch_local_passthrough,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} WAN FFN chunking tests")


if __name__ == "__main__":
    main()
