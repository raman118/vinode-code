"""Local Gradio demo for classical and optional trained YOLO inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from colony_counting.classical import (
    ClassicalResult,
    count_colonies,
    detect_plate_circle,
    load_classical_config,
)


def process_image(
    image_rgb: np.ndarray | None,
    method: str,
    confidence: float,
    *,
    config_path: Path = Path("configs/classical.yaml"),
    checkpoint: Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str, str]:
    """Run selected real inference and return two overlays, counts, latency, and any error."""

    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None, None, "No image uploaded", "Not measured", "Upload a plate image to start."
    if method not in {"Classical", "YOLO", "Side by side"}:
        return None, None, "Invalid method", "Not measured", f"Unsupported method: {method}"
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    classical_overlay = None
    yolo_overlay = None
    count_lines: list[str] = []
    latency_lines: list[str] = []
    errors: list[str] = []

    if method in {"Classical", "Side by side"}:
        try:
            config = load_classical_config(config_path)
            started = time.perf_counter()
            result: ClassicalResult = count_colonies(image_bgr, config)
            elapsed = time.perf_counter() - started
            classical_overlay = cv2.cvtColor(result.stages["overlay"], cv2.COLOR_BGR2RGB)
            count_lines.append(f"Classical: {result.count}")
            latency_lines.append(f"Classical: {elapsed:.4f} s")
        except Exception as exc:  # noqa: BLE001 - report runtime failures in the UI.
            count_lines.append("Classical: failed")
            latency_lines.append("Classical: unavailable")
            errors.append(f"Classical inference failed: {exc}")

    if method in {"YOLO", "Side by side"}:
        if checkpoint is None or not Path(checkpoint).is_file():
            count_lines.append("YOLO: checkpoint required")
            latency_lines.append("YOLO: not measured")
            errors.append(
                "No trained YOLO checkpoint is available. Train a detector and relaunch with --checkpoint path\\to\\best.pt."
            )
        else:
            try:
                from colony_counting.detection.yolo import infer_yolo

                config = load_classical_config(config_path)
                plate_circle = detect_plate_circle(image_bgr, config)
                started = time.perf_counter()
                detections = infer_yolo(
                    image=image_bgr,
                    checkpoint=checkpoint,
                    confidence=confidence,
                    plate_circle=plate_circle,
                )
                elapsed = time.perf_counter() - started
                yolo_overlay = image_bgr.copy()
                for detection in detections:
                    x1, y1, x2, y2 = map(
                        round, (detection.x1, detection.y1, detection.x2, detection.y2)
                    )
                    cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), (20, 40, 240), 2)
                yolo_overlay = cv2.cvtColor(yolo_overlay, cv2.COLOR_BGR2RGB)
                count_lines.append(f"YOLO: {len(detections)}")
                latency_lines.append(f"YOLO: {elapsed:.4f} s")
            except Exception as exc:  # noqa: BLE001 - report model loading/inference errors in the UI.
                count_lines.append("YOLO: failed")
                latency_lines.append("YOLO: unavailable")
                errors.append(f"YOLO model loading or inference failed: {exc}")

    return (
        classical_overlay,
        yolo_overlay,
        "\n".join(count_lines),
        "\n".join(latency_lines),
        "\n".join(errors) if errors else "No errors.",
    )


def build_app(*, config_path: Path, checkpoint: Path | None) -> Any:
    """Build the local Blocks interface. Import Gradio only when the app is requested."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The local demo requires the app extra: pip install -e '.[app]'"
        ) from exc

    def callback(
        image: np.ndarray | None, method: str, confidence: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, str, str, str]:
        return process_image(
            image,
            method,
            confidence,
            config_path=config_path,
            checkpoint=checkpoint,
        )

    with gr.Blocks(title="Microbial Colony Counter") as demo:
        gr.Markdown(
            "# Microbial Colony Counter\n"
            "Local comparison demo. Counts depend on image quality and method settings; "
            "no result is a substitute for manual scientific review."
        )
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload plate image")
            with gr.Column():
                method = gr.Radio(
                    choices=["Classical", "YOLO", "Side by side"],
                    value="Classical",
                    label="Method",
                )
                confidence = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=0.25,
                    step=0.01,
                    label="YOLO confidence threshold",
                )
                submit = gr.Button("Count colonies", variant="primary")
        with gr.Row():
            classical_output = gr.Image(label="Classical overlay")
            yolo_output = gr.Image(label="YOLO overlay")
        counts = gr.Textbox(label="Counts", lines=2, interactive=False)
        latency = gr.Textbox(label="Inference latency", lines=2, interactive=False)
        errors = gr.Textbox(label="Model and inference status", lines=3, interactive=False)
        submit.click(
            fn=callback,
            inputs=[input_image, method, confidence],
            outputs=[classical_output, yolo_output, counts, latency, errors],
        )
    return demo


def launch_app(
    *,
    config_path: Path = Path("configs/classical.yaml"),
    checkpoint: Path | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> None:
    """Launch a local-only demo without creating a public share link."""

    app = build_app(config_path=config_path, checkpoint=checkpoint)
    app.launch(server_name=server_name, server_port=server_port, share=False)
