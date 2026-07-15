"""Image Enhancement Web App - AI-powered upscaling, denoise, deblur, and color correction."""

import io
import logging
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageFilter

from esrgan_model import RealESRGANUpscaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Enhancer",
    description="AI-powered image enhancement: denoise, deblur, 4x AI upscale, and color correct.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def preload_model():
    """Log that the model will load on first request."""
    logger.info("Real-ESRGAN will load on first enhance request (uses MPS acceleration).")

# --- Real-ESRGAN Model Setup ---
ESRGAN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "RealESRGAN_x4plus.pth")
upsampler = None


def get_upsampler():
    """Lazy-load the Real-ESRGAN model."""
    global upsampler
    if upsampler is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        upsampler = RealESRGANUpscaler(
            model_path=ESRGAN_MODEL_PATH,
            device=device,
            tile_size=512,
            tile_pad=10,
        )
        logger.info(f"Real-ESRGAN model loaded on device: {device}")
    return upsampler


# --- Enhancement Functions ---


def denoise_image(img_array: np.ndarray, strength: int = 10) -> np.ndarray:
    """Remove grain/noise using Non-Local Means Denoising."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(img_array, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(img_array, None, strength, 7, 21)


def bilateral_denoise(img_array: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral filter for edge-preserving noise reduction."""
    return cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)


def sharpen_image(img: Image.Image, passes: int = 2) -> Image.Image:
    """Reduce blur by sharpening the image multiple passes."""
    for _ in range(passes):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def unsharp_mask(img_array: np.ndarray, sigma: float = 1.5, strength: float = 2.0) -> np.ndarray:
    """Apply unsharp mask for targeted deblurring."""
    blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_array, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def detail_enhance(img_array: np.ndarray) -> np.ndarray:
    """OpenCV detail enhancement for recovering fine textures."""
    return cv2.detailEnhance(img_array, sigma_s=10, sigma_r=0.15)


def ai_upscale(img_array: np.ndarray, target_scale: float = 4.0) -> np.ndarray:
    """AI upscale using Real-ESRGAN (generates new detail, not just interpolation)."""
    sr = get_upsampler()
    return sr.enhance(img_array, outscale=target_scale)


def fallback_upscale(img: Image.Image, scale_factor: float = 4.0) -> Image.Image:
    """Fallback Lanczos upscale if AI model isn't available."""
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    return img.resize((new_width, new_height), Image.LANCZOS)


def boost_saturation(img: Image.Image, factor: float = 1.08) -> Image.Image:
    """Increase color saturation."""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def vibrance_boost(img_array: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Photoshop-style vibrance: boosts muted colors more, preserves skin tones.

    Unlike saturation which is uniform, vibrance targets under-saturated pixels
    so already-vibrant areas (like skin) don't get over-cooked.
    """
    img_float = img_array.astype(np.float32) / 255.0

    # Convert to HSV to analyze saturation per pixel
    hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Boost inversely proportional to current saturation
    # Low saturation pixels get more boost, high saturation pixels get less
    boost_mask = (1.0 - s) * strength
    s_new = np.clip(s + boost_mask * s, 0, 1)

    hsv_new = cv2.merge([h, s_new, v])
    result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB)
    return (result * 255).clip(0, 255).astype(np.uint8)


def auto_levels(img_array: np.ndarray, shadow_clip: float = 0.5, highlight_clip: float = 0.5) -> np.ndarray:
    """Photoshop-style auto levels: stretches histogram per channel.

    Clips a percentage of shadow/highlight pixels to expand tonal range,
    similar to Image > Auto Levels in Photoshop.
    """
    result = np.zeros_like(img_array)

    for c in range(3):
        channel = img_array[:, :, c]
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        total_pixels = channel.size

        # Find shadow clip point
        shadow_threshold = total_pixels * (shadow_clip / 100.0)
        cumsum = 0
        low = 0
        for i in range(256):
            cumsum += hist[i]
            if cumsum >= shadow_threshold:
                low = i
                break

        # Find highlight clip point
        highlight_threshold = total_pixels * (highlight_clip / 100.0)
        cumsum = 0
        high = 255
        for i in range(255, -1, -1):
            cumsum += hist[i]
            if cumsum >= highlight_threshold:
                high = i
                break

        # Stretch the range
        if high > low:
            scale = 255.0 / (high - low)
            result[:, :, c] = np.clip((channel.astype(np.float32) - low) * scale, 0, 255).astype(np.uint8)
        else:
            result[:, :, c] = channel

    return result


def midtone_boost(img_array: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """Lift midtones using gamma correction (gamma < 1 = brighter mids)."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_array, table)


def warm_tone(img_array: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Add subtle warmth by slightly boosting red channel and reducing blue."""
    result = img_array.astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] + strength, 0, 255)  # R
    result[:, :, 2] = np.clip(result[:, :, 2] - strength * 0.5, 0, 255)  # B
    return result.astype(np.uint8)


def auto_contrast(img: Image.Image, factor: float = 1.08) -> Image.Image:
    """Apply auto-contrast for better tonal range."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def auto_brightness(img: Image.Image, factor: float = 1.03) -> Image.Image:
    """Slight brightness lift for dark screenshots."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


@app.get("/health")
async def health_check():
    model_loaded = upsampler is not None
    return {
        "status": "healthy",
        "service": "image-enhancer",
        "version": "2.0.0",
        "ai_model_loaded": model_loaded,
    }


@app.post("/enhance")
async def enhance_image(
    image: UploadFile = File(..., description="Image file to enhance (PNG, JPEG, WebP)"),
    denoise_strength: int = Form(6, description="Denoising strength (1-30, default 6)"),
    sharpen_passes: int = Form(1, description="Sharpening passes (1-5, default 1)"),
    upscale_factor: float = Form(4.0, description="Upscale factor (1.0-4.0, default 4.0)"),
    saturation_boost: float = Form(1.08, description="Saturation multiplier"),
    brightness: float = Form(1.03, description="Brightness multiplier"),
    contrast: float = Form(1.08, description="Contrast multiplier"),
    warmth: int = Form(8, description="Warmth shift (0-30)"),
    apply_denoise: bool = Form(False, description="Apply denoising (only for grainy/noisy images)"),
    apply_sharpen: bool = Form(True, description="Apply sharpening/deblur"),
    apply_upscale: bool = Form(True, description="Apply AI upscaling"),
    apply_color: bool = Form(True, description="Apply color correction"),
    output_format: str = Form("png", description="Output format: png or jpg"),
    jpg_quality: int = Form(92, description="JPEG quality (1-100, default 92)"),
):
    """Enhance an image through the AI-powered pipeline.

    Steps:
    1. Denoise (bilateral + non-local means for grain/noise removal)
    2. Sharpen/Deblur (adaptive sharpening + unsharp mask + detail enhance)
    3. AI Upscale (Real-ESRGAN 4x - generates new detail like Adobe Firefly)
    4. Color correction (saturation boost + contrast + brightness)
    """
    # Clamp parameters to safe ranges
    denoise_strength = max(1, min(30, denoise_strength))
    sharpen_passes = max(1, min(5, sharpen_passes))
    upscale_factor = max(1.0, min(4.0, upscale_factor))
    saturation_boost = max(1.0, min(1.20, saturation_boost))

    # Read and decode image
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    original_size = img.size
    logger.info(f"Received image: {original_size[0]}x{original_size[1]}, size={len(data)/1024:.1f}KB")

    # Cap input at 2048px on the long edge for processing speed
    # (the AI upscaler will bring it back up to 4K)
    max_input_dim = 2048
    w, h = img.size
    if max(w, h) > max_input_dim:
        scale = max_input_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Downscaled input for processing: {w}x{h} → {new_w}x{new_h}")

    # Step 1: Denoise (light single-pass, preserves texture)
    if apply_denoise:
        img_array = np.array(img)
        img_array = denoise_image(img_array, strength=denoise_strength)
        img = Image.fromarray(img_array)
        logger.info(f"Denoised with strength={denoise_strength}")

    # Step 2: Sharpen / Deblur (gentle unsharp mask only)
    if apply_sharpen:
        img_array = np.array(img)
        img_array = unsharp_mask(img_array, sigma=1.0, strength=0.5)
        img = Image.fromarray(img_array)
        if sharpen_passes > 0:
            img = sharpen_image(img, passes=min(sharpen_passes, 1))
        logger.info(f"Sharpened (light unsharp mask + {min(sharpen_passes, 1)} pass)")

    # Step 3: AI Upscale with Real-ESRGAN
    if apply_upscale:
        try:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            long_edge = max(w, h)

            # Target 4K (3840px) on the long edge, but don't downscale
            target_long_edge = 3840
            if long_edge >= target_long_edge:
                # Already 4K or larger — skip AI upscale, no benefit
                logger.info(f"Image already {w}x{h} (≥4K), skipping AI upscale")
            else:
                # Calculate actual scale needed to reach 4K
                needed_scale = min(target_long_edge / long_edge, upscale_factor)
                # Real-ESRGAN natively does 4x, then we resize to target
                img_array = ai_upscale(img_array, target_scale=needed_scale)
                img = Image.fromarray(img_array)
                logger.info(f"AI upscaled {needed_scale:.1f}x → {img.width}x{img.height} (Real-ESRGAN)")
        except Exception as e:
            logger.warning(f"AI upscale failed ({e}), falling back to Lanczos")
            img = fallback_upscale(img, scale_factor=upscale_factor)
            logger.info(f"Fallback upscaled {upscale_factor}x → {img.width}x{img.height}")

    # Step 4: Color correction (matches live preview sliders exactly)
    if apply_color:
        # Brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        # Contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        # Saturation
        if saturation_boost != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation_boost)
        # Warmth (shift R up, B down)
        if warmth > 0:
            img_array = np.array(img, dtype=np.float32)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + warmth, 0, 255)  # R
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] - warmth * 0.4, 0, 255)  # B
            img = Image.fromarray(img_array.astype(np.uint8))
        logger.info(f"Color corrected: brightness={brightness}, contrast={contrast}, sat={saturation_boost}, warmth={warmth}")

    # Encode output in selected format
    output_format = output_format.lower().strip()
    if output_format not in ("png", "jpg", "jpeg"):
        output_format = "png"

    output_buffer = io.BytesIO()
    if output_format in ("jpg", "jpeg"):
        jpg_quality = max(1, min(100, jpg_quality))
        img.save(output_buffer, format="JPEG", quality=jpg_quality, optimize=True)
        media_type = "image/jpeg"
        file_ext = "jpg"
    else:
        img.save(output_buffer, format="PNG", optimize=True)
        media_type = "image/png"
        file_ext = "png"

    output_bytes = output_buffer.getvalue()

    logger.info(
        f"Enhancement complete: {original_size[0]}x{original_size[1]} → {img.width}x{img.height}, "
        f"format={file_ext}, output size={len(output_bytes)/1024:.1f}KB"
    )

    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "X-Original-Width": str(original_size[0]),
            "X-Original-Height": str(original_size[1]),
            "X-Enhanced-Width": str(img.width),
            "X-Enhanced-Height": str(img.height),
            "X-Output-Format": file_ext,
            "Content-Disposition": f"attachment; filename=enhanced_image.{file_ext}",
        },
    )


# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
