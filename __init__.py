import torch
import nvvfx
from enum import Enum
from typing import TypedDict
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


def _resolve_output_dimensions(
    input_width: int,
    input_height: int,
    resize_type: "RTXImageSuperResolution.UpscaleTypedDict",
) -> tuple[int, int]:
    """Compute 8-pixel-aligned output dimensions from the resize config."""
    selected_type = resize_type["resize_type"]
    if selected_type == UpscaleType.SCALE_BY:
        scale = resize_type["scale"]
        output_width = int(input_width * scale)
        output_height = int(input_height * scale)
    elif selected_type == UpscaleType.TARGET_DIMENSIONS:
        output_width = resize_type["width"]
        output_height = resize_type["height"]
    else:
        raise ValueError(f"Unsupported resize type: {selected_type}")

    output_width = max(8, round(output_width / 8) * 8)
    output_height = max(8, round(output_height / 8) * 8)
    return output_width, output_height


def _get_quality_level(quality: str):
    """Map quality string to nvvfx QualityLevel enum."""
    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    return quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)


# Shared input/output schema fragments
_RESIZE_TYPE_INPUT = io.DynamicCombo.Input(
    "resize_type",
    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
    options=[
        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
        ]),
        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels.")
        ])
    ],
)

_QUALITY_INPUT = io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA")


class RTXImageSuperResolution(io.ComfyNode):
    """Upscale a single image using NVIDIA RTX Video Super Resolution."""

    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXImageSuperResolution",
            display_name="RTX Image Super Resolution",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution"],
            inputs=[
                io.Image.Input("image"),
                _RESIZE_TYPE_INPUT,
                _QUALITY_INPUT,
            ],
            outputs=[
                io.Image.Output("upscaled_image"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, h, w, _ = image.shape
        output_width, output_height = _resolve_output_dimensions(w, h, resize_type)
        selected_quality = _get_quality_level(quality)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            input_cuda = image[0].cuda().permute(2, 0, 1).contiguous()
            dlpack_out = sr.run(input_cuda).image
            output = torch.from_dlpack(dlpack_out).clone()

        result = output.permute(1, 2, 0).unsqueeze(0).cpu()
        return io.NodeOutput(result)


class RTXVideoSuperResolution(io.ComfyNode):
    """Upscale a batch of video frames using NVIDIA RTX Video Super Resolution."""

    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr", "video"],
            inputs=[
                io.Image.Input("images"),
                _RESIZE_TYPE_INPUT,
                _QUALITY_INPUT,
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, h, w, _ = images.shape
        output_width, output_height = _resolve_output_dimensions(w, h, resize_type)
        selected_quality = _get_quality_level(quality)

        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        upscaled_batches = []

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]

                batch_cuda = batch.cuda().permute(0, 3, 1, 2).contiguous()

                batch_outputs = []

                for j in range(batch_cuda.shape[0]):
                    input_frame = batch_cuda[j]
                    dlpack_out = sr.run(input_frame).image
                    output = torch.from_dlpack(dlpack_out).clone()
                    batch_outputs.append(output)

                batch_out_tensor = torch.stack(batch_outputs, dim=0)

                batch_out_tensor = batch_out_tensor.permute(0, 2, 3, 1).cpu()
                upscaled_batches.append(batch_out_tensor)

        final_images = torch.cat(upscaled_batches, dim=0)

        return io.NodeOutput(final_images)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXImageSuperResolution,
            RTXVideoSuperResolution,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()

# hack so registry picks up the node name
if False:
    NODE_CLASS_MAPPINGS = {
        "RTXImageSuperResolution": RTXImageSuperResolution,
        "RTXVideoSuperResolution": RTXVideoSuperResolution,
    }
