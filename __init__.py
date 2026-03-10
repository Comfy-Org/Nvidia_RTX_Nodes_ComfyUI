import torch
import nvvfx
from enum import Enum
from typing import TypedDict
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class RTXVideoSuperResolution(io.ComfyNode):
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
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
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
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16

        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)

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


class RTXDenoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXDenoise",
            display_name="RTX Denoise",
            category="image/postprocessing",
            search_aliases=["rtx", "nvidia", "denoise", "noise removal"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("strength", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH"),
            ],
            outputs=[
                io.Image.Output("denoised_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, strength: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        output_width = max(8, round(w / 8) * 8)
        output_height = max(8, round(h / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        strength_mapping = {
            "LOW": nvvfx.effects.QualityLevel.DENOISE_LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.DENOISE_MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.DENOISE_HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.DENOISE_ULTRA,
        }
        selected_quality = strength_mapping.get(strength, nvvfx.effects.QualityLevel.DENOISE_HIGH)

        output_batches = []

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
                output_batches.append(batch_out_tensor)

        final_images = torch.cat(output_batches, dim=0)
        return io.NodeOutput(final_images)


class RTXDeblur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXDeblur",
            display_name="RTX Deblur",
            category="image/postprocessing",
            search_aliases=["rtx", "nvidia", "deblur", "sharpen", "motion blur"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("strength", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH"),
            ],
            outputs=[
                io.Image.Output("deblurred_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, strength: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        output_width = max(8, round(w / 8) * 8)
        output_height = max(8, round(h / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        strength_mapping = {
            "LOW": nvvfx.effects.QualityLevel.DEBLUR_LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.DEBLUR_MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.DEBLUR_HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.DEBLUR_ULTRA,
        }
        selected_quality = strength_mapping.get(strength, nvvfx.effects.QualityLevel.DEBLUR_HIGH)

        output_batches = []

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
                output_batches.append(batch_out_tensor)

        final_images = torch.cat(output_batches, dim=0)
        return io.NodeOutput(final_images)


class RTXHighBitrateUpscale(io.ComfyNode):
    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXHighBitrateUpscale",
            display_name="RTX High Bitrate Upscale",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "high bitrate", "clean", "lossless"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
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
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, resize_type: UpscaleTypedDict, quality: str) -> io.NodeOutput:
        _, h, w, _ = images.shape

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.HIGHBITRATE_LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.HIGHBITRATE_MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGHBITRATE_HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.HIGHBITRATE_ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGHBITRATE_HIGH)

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
            RTXVideoSuperResolution,
            RTXDenoise,
            RTXDeblur,
            RTXHighBitrateUpscale,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()
