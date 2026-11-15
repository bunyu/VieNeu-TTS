import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")
import soundfile as sf
import tempfile
from vieneu import Vieneu
import os
import time
import numpy as np
import queue
import threading
from pathlib import Path
import yaml
import uuid
from vieneu_utils.core_utils import join_audio_chunks, env_bool, get_silence_duration_v2, gaps_to_silence, trust_remote_code_enabled
from vieneu_utils.phonemize_text import phonemize_to_chunks, normalize_to_chunks, normalize_to_chunks_v3, normalize_to_chunks_v3_with_gaps
# PuncNormalizer = sea_g2p.Normalizer luôn bật punc_norm=True.
from vieneu_utils.phonemize_text import PuncNormalizer as Normalizer
import gc

# PDF text extraction
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    fitz = None

from apps.srt_speech import srt_to_speech
from apps.user_voices import (
    load_user_voices, save_user_voice, delete_user_voice, list_user_voices, supports_saving,
)
from apps.ui_utils import (
    _format_duration,
    _split_estimate_status,
    wrap_with_estimate,
    cleanup_gpu_memory,
    get_ref_text_cached,
    validate_audio_duration,
    on_custom_id_change
)
from apps.ui_constants import (
    theme,
    css,
    head_html,
    DEFAULT_TEXT_GPU,
    DEFAULT_TEXT_TURBO,
    DEFAULT_TEXT_V3
)

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})

# Refilter and Simplify Configs per requirements
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}


def _watermark_available() -> bool:
    """True only when the Resemble `resemble-perth` watermarker can be used.

    The PyPI package literally named `perth` is an unrelated threading helper;
    the SDK's `_init_watermarker` falls back to *no* watermark in that case, so
    the UI must not claim the audio is watermarked (issue #191).
    """
    try:
        import perth  # provided by `resemble-perth` (pip install vieneu[watermark])
        return getattr(perth, "PerthImplicitWatermarker", None) is not None
    except Exception:
        return False


# VieNeu-TTS v3 Turbo — the DEFAULT on both CPU and GPU (first entry = default in the UI).
# CPU runs the fp32 ONNX graphs (maximum quality); GPU runs PyTorch.
filtered_backbones["VieNeu-TTS-v3-Turbo"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "precision": "fp32",
    "supports_streaming": False,
    "description": (
        "🆕 v3 Turbo — 48kHz, bản mặc định. Giọng mặc định dùng speaker token (ổn định hơn); Voice Cloning "
        "clone từ audio mẫu; tag cảm xúc [cười]/[hắng giọng]/[thở dài] (thử nghiệm)."
        + ("" if HAS_GPU else " Trên CPU chạy ONNX fp32 (chất lượng tối đa).")
    )
}
if not HAS_GPU:
    filtered_backbones["VieNeu-TTS-v3-Turbo (int8)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        "precision": "int8",
        "supports_streaming": False,
        "description": "v3 Turbo (int8) — 48kHz, backbone nén int8: nhanh hơn bản fp32 trên CPU có AVX-512/AVX-VNNI (CPU cũ không có VNNI có thể bị méo tiếng). Cùng giọng, cloning và tag cảm xúc như bản mặc định."
    }
# VieNeu-TTS v3 Nano (PREVIEW) — ONNX/CPU only. NOT the default: a 48M-param flow model for
# edge devices (Android, weak CPUs). Listed last so Turbo stays the first/default entry.
filtered_backbones["VieNeu-TTS-v3-Nano (preview)"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Nano",
    "supports_streaming": False,
    "description": (
        "🪶 v3 Nano (PREVIEW, đang thử nghiệm) — 24kHz, model flow 48M tham số, chạy RẤT NHANH, dành cho "
        "edge device (Android) hoặc CPU yếu. Chất lượng KÉM HƠN NHIỀU so với v3 Turbo, nhất là tiếng Anh và "
        "câu song ngữ; chỉ 6 giọng có sẵn, KHÔNG clone giọng; còn nhiều thiếu sót. Chỉ dùng khi thực sự cần "
        "tốc độ hoặc deploy trên điện thoại."
    )
}

# Models fine-tuned on this PC for one voice (finetune/output/<voice>/merged): PyTorch, so GPU only.
if HAS_GPU:
    try:
        from apps.finetune_studio import merged_models, model_label
        for _run, _folder in merged_models():
            filtered_backbones[model_label(_run)] = {
                "repo": _folder, "precision": "fp32", "supports_streaming": False,
                "description": f"🎓 v3 Turbo đã fine-tune cho giọng “{_run}”. Sau khi nạp, chọn giọng “{_run}” trong danh sách giọng mẫu.",
            }
    except Exception as _e:  # noqa: BLE001
        print(f"⚠️ Không đọc được các model fine-tune: {_e}")

# GPU-only extras. On GPU the default is VieNeu-TTS-v2 (GPU); v3 Turbo stays the
# default (and only option) on CPU machines (the v2/v1 GGUF CPU builds were removed).
if HAS_GPU:
    filtered_backbones["VieNeu-TTS-v2 (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 2 - hỗ trợ song ngữ (Anh-Việt) và chế độ podcast"
    }
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 1 - ổn định, production-ready"
    }

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "NeuCodec (ONNX)": {
        "repo": "neuphonic/neucodec-onnx-decoder-int8",
        "description": "Codec siêu nhẹ, tối ưu cho CPU (ONNX)",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False
PRESET_VOICES_CACHE = []  # List of all voices (tuples or strings)
CONV_VOICES_CACHE = []    # Filtered list for conversation (podcast=True)


def _sort_voices(tts, voices):
    """Dropdown order: editors' picks (``featured`` 1..N in the voices JSON, already
    ⭐-labelled by the SDK) first in that order, then everything else A-Z."""
    presets = getattr(tts, "_preset_voices", {}) or {}

    def _key(v):
        label, v_id = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
        rank = presets.get(v_id, {}).get("featured") if isinstance(presets.get(v_id), dict) else None
        return (rank is None, rank or 0, str(label))

    voices.sort(key=_key)
MAX_SPEAKERS = 8          # Max concurrent speakers in conversation tab

# Normalizer (module-level singleton)
_text_normalizer = Normalizer()

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append("MPS")
        elif torch.cuda.is_available():
            devices.append("CUDA")
    except ImportError:
        pass

    return devices

def _supports_cloning(backbone_choice: str) -> bool:
    """Voice Cloning availability by model.

    v3 Turbo and v3 Nano clone directly from a sample audio (Nano fetches its
    cloning graphs on first use); VieNeu-TTS-v2 (GPU) clones from audio plus a
    reference transcript. v1 and the CPU v2 builds stay preset-only.
    """
    c = (backbone_choice or "").lower()
    return "v3" in c or c == "vieneu-tts-v2 (gpu)"

def get_model_status_message() -> str:
    """Reconstruct status message from global state"""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return "⏳ Chưa tải model."
    
    if "v2-Turbo" in (current_backbone or ""):
        backend_name = "⚡ Turbo (v2)"
    elif using_lmdeploy:
        backend_name = "🚀 LMDeploy (Optimized)"
    else:
        backend_name = "📦 Standard"
    
    # We don't track the exact device strings perfectly in global state, so we estimate
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except:
        has_mps = has_cuda = False

    device_info = "GPU (CUDA)" if (using_lmdeploy or "CUDA" in (current_backbone or "")) else ("MPS (Metal)" if has_mps else "Auto")
    
    if "v2-Turbo" in (current_backbone or ""):
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"
    elif "ONNX" in (current_codec or ""):
        codec_device = "CPU"
    else:
        codec_device = "GPU/MPS" if (has_cuda or has_mps) else "CPU"

    preencoded_note = ""    
    opt_info = ""
    if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
        stats = tts.get_optimization_stats()
        opt_info = (
            f"\n\n🔧 Tối ưu hóa:"
            f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
            f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
            f"\n  • Reference Cache: {stats['cached_references']} voices"
            f"\n  • Prefix Caching: ❌"
        )

    return (
        f"✅ Model đã tải thành công!\n\n"
        f"🔧 Backend: {backend_name}\n"
        f" Parrot: {current_backbone} on {device_info}\n"
        f"🎵 Codec: {current_codec} on {codec_device}{preencoded_note}{opt_info}"
    )

_autoload_started = False


def autoload_model(*load_inputs):
    """Load the default model as soon as the page opens (VIENEU_AUTOLOAD=1, set by run_windows.bat).

    The model files stay in the Hugging Face cache (HF_HOME) after the first download; what each start
    still needs is reading them into memory, and this does it without a click on the load button.
    """
    global _autoload_started
    if _autoload_started or model_loaded or not env_bool("VIENEU_AUTOLOAD", default=False):
        return
    _autoload_started = True
    yield from load_model(*load_inputs)


LAB_MAX = 6  # clone candidates shown side by side
VOICE_SAMPLE_TEXT = "Xin chào, tôi sẽ đọc sách cho bạn nghe bằng giọng này. Chúc bạn có những phút giây thật thư giãn."
_voice_samples: dict = {}


def voice_sample(voice_choice):
    """A short sentence in the chosen voice, kept for the session so a second listen is instant."""
    if not model_loaded or tts is None:
        raise gr.Error("Vui lòng nạp model trước!")
    voice_id = resolve_voice_id(voice_choice) if voice_choice else None
    key = (current_backbone, voice_id)
    if key not in _voice_samples:
        kwargs = {"voice": tts.get_preset_voice(voice_id)} if voice_id else {}
        _voice_samples[key] = np.asarray(tts.infer(VOICE_SAMPLE_TEXT, **kwargs), dtype=np.float32).reshape(-1)
    return tts.sample_rate, _voice_samples[key]


def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=model_loaded), # btn_generate_conv
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    # GGUF, v2-Turbo và v3 Turbo đều KHÔNG dùng LMDeploy (v3 là PyTorch, có engine riêng).
    bc = backbone_choice.lower()
    if "gguf" in bc or "v2-turbo" in bc or "v3" in bc:
        return False
    
    try:
        import torch
        if device_choice == "Auto":
            has_gpu = torch.cuda.is_available()
        elif device_choice == "CUDA":
            has_gpu = torch.cuda.is_available()
        else:
            has_gpu = False
        return has_gpu
    except ImportError:
        return False

def load_model(backbone_choice: str, codec_choice: str, device_choice: str, 
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "", 
               custom_hf_token: str = ""):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False # Ensure we don't try to use a half-loaded model
    
    # Helper for slot updates (initially no change)
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    yield (
        "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
        gr.update(interactive=False), # btn_generate
        gr.update(interactive=False), # btn_generate_conv
        gr.update(interactive=False), # btn_load
        gr.update(interactive=False), # btn_stop
        gr.update(), # voice_select
        gr.update(), # tab_custom
        gr.update(), # conv_tab
        *slot_no_updates
    )
    
    try:
        # Cleanup before loading new model
        if tts is not None:
            tts = None # Reset instead of del to avoid NameError if load fails
            cleanup_gpu_memory()
        
        # Prepare Backbone Config/Repo
        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), # tab_custom
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                return

            # Check if it is a LoRA to merge
            if "lora" in custom_model_id.lower():
                # Merging mode
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), # conv_tab
                        *slot_no_updates
                    )
                    return
                
                base_config = BACKBONE_CONFIGS[custom_base_model]
                backbone_config = {
                    "repo": base_config["repo"], # Load base first
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                # Normal custom model
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False, # Assume false for unknown
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]
            
        codec_config = CODEC_CONFIGS[codec_choice]
        use_lmdeploy = False
        
        # Override LMDeploy if custom
        if custom_loading:
             if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                 # GGUF must use Standard/Turbo backend
                 use_lmdeploy = False
             elif is_merged_lora:
                 # LoRA can use LMDeploy if we merge first (checked logic below) or Standard
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 # Full custom model (e.g. finetune). LMDeploy reads the repo itself and
                 # older releases always pass trust_remote_code=True, so a typed-in repo
                 # takes that path only with VIENEU_TRUST_REMOTE_CODE=1; the Standard
                 # backend never runs the repo's code.
                 use_lmdeploy = (force_lmdeploy and trust_remote_code_enabled()
                                 and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice))
        # Use LMDeploy only if Force LMDeploy is set and the model is compatible
        # NOTE: For VieNeu-v2-Turbo, we handle LMDeploy inside TurboGPUVieNeuTTS class, 
        # so we set use_lmdeploy = False here to avoid generic FastVieNeuTTS loading.
        # NOTE: For custom_loading, the block above already decided use_lmdeploy correctly
        # (e.g. False for GGUF repos). Do NOT override that decision here.
        if "v2-Turbo" in backbone_choice or "v3" in backbone_choice.lower():
             # v2-Turbo có LMDeploy riêng trong class; v3 Turbo là PyTorch (engine riêng,
             # không bao giờ dùng LMDeploy) — bỏ qua FastVieNeuTTS generic.
             should_use_generic_fast = False
        elif custom_loading:
             should_use_generic_fast = False  # already handled above per repo name
        else:
             should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
             
        if should_use_generic_fast:
            use_lmdeploy = True
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                try:
                    import torch
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    codec_device = "cpu"
            
            # Special handling for Custom LoRA + LMDeploy -> Merge & Save
            target_backbone_repo = backbone_config["repo"]
            
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)
                
                # Check if already merged (and voices.json exists)
                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    if os.path.exists(cache_dir):
                        print("   ⚠️ Detected incomplete cache, rebuilding...")
                    yield (
                         f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), # tab_custom
                         gr.update(), # conv_tab
                         *slot_no_updates
                    )
                    
                    try:
                        # Use GPU for merging if available for speed
                        # We use the Base Model specified
                        from vieneu.standard import VieNeuTTS
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        print(f"   • Loading base: {base_repo} ({merge_device})")
                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device, 
                            codec_repo=codec_config["repo"],
                            codec_device="cpu", # Codec unused for merging, keep on CPU
                            hf_token=custom_hf_token
                        )
                        
                        print(f"   • Loading Adapter: {custom_model_id}")
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                        
                        print(f"   • Merging...")
                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                        
                        print(f"   • Saving to cache: {cache_dir}")
                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)
                        
                        # Fix for LMDeploy: Explicitly save legacy tokenizer files (vocab.json, merges.txt)
                        # because LMDeploy/Transformers might default to slow tokenizer if fast one has issues,
                        # and save_pretrained on fast tokenizer sometimes omits legacy files.
                        try:
                            print("   • Ensuring legacy tokenizer files...")
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not save slow tokenizer files: {e}")

                        # Save voices.json to cache directory so FastVieNeuTTS can find it
                        print(f"   • Saving voices definition...")
                        import json
                        voices_json_path = os.path.join(cache_dir, "voices.json")
                        voices_content = {
                             "meta": { "note": "Automatically generated during LoRA merge" },
                             "default_voice": temp_tts._default_voice,
                             "presets": temp_tts._preset_voices
                        }
                        with open(voices_json_path, 'w', encoding='utf-8') as f:
                             json.dump(voices_content, f, ensure_ascii=False, indent=2)

                        del temp_tts
                        cleanup_gpu_memory()
                        print("   ✅ Merge & Save successfully!")
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {target_backbone_repo} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: Enabled")
            
            try:
                from vieneu.fast import FastVieNeuTTS
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=False,
                    enable_triton=True,
                    hf_token=custom_hf_token
                )
                using_lmdeploy = True
                
                # Legacy caching removed
                print(f"   ✅ Optimized backend initialized")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"
                
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower

                if is_gguf_backbone:
                    # GGUF backbones (llama-cpp-python): Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal via n_gpu_layers
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    # PyTorch backbones (Standard, Turbo GPU): use native torch device
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        codec_device = "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                # Only Llama-cpp (GGUF) uses the 'gpu' string for CUDA
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            if "v3-Nano" in backbone_choice:
                # VieNeu v3 Nano: ONNX/CPU only (torch-free), preset voices only. Chosen
                # explicitly by the user for weak CPUs — never auto-selected.
                print("   🪶 Mode: v3 Nano (ONNX/CPU, 24 kHz, preset voices only)")
                tts = Vieneu(
                    mode="v3nano",
                    backbone_repo=backbone_config["repo"],
                    hf_token=custom_hf_token,
                )
            elif "v3-Turbo" in backbone_choice:
                # VieNeu v3 Turbo. CPU → ONNX Runtime; GPU → PyTorch. The backend is
                # auto-selected from the device inside Vieneu(mode="v3turbo"); ONNX
                # graphs are fetched from the model repo's onnx/ subfolder.
                print("   🆕 Mode: v3 Turbo (CPU=ONNX / GPU=PyTorch)")
                # Map the app's device string to what the v3 engine understands.
                v3_device = "cpu" if str(backbone_device).lower() == "cpu" else "auto"
                # precision: "int8" (mặc định, subfolder onnx_int8) | "fp32" (onnx_update).
                # Chỉ ảnh hưởng đường CPU/ONNX; trên GPU dùng PyTorch nên bỏ qua.
                v3_precision = backbone_config.get("precision", "int8")
                print(f"   🎚️  Precision: {v3_precision}")
                # On a CUDA card, pick a dtype and batch size that suit it (GTX 10-series:
                # float32, since its half precision is very slow and it has no bfloat16).
                from vieneu.gpu_support import turbo_gpu_kwargs
                v3_gpu = turbo_gpu_kwargs() if v3_device != "cpu" else {}
                if v3_gpu:
                    print(f"   🖥️  GPU settings: {v3_gpu}")
                tts = Vieneu(
                    mode="v3turbo",
                    backbone_repo=backbone_config["repo"],
                    device=v3_device,
                    precision=v3_precision,
                    hf_token=custom_hf_token,
                    **{k: v for k, v in v3_gpu.items() if k != "device"},
                )
            elif "v2-Turbo" in backbone_choice:
                # VieNeu v2 Turbo uses the dedicated backend
                print("   ⚡ Mode: Turbo")
                mode = "turbo_gpu" if "GPU" in backbone_choice else "turbo"
                tts = Vieneu(
                    mode=mode,
                    backbone_repo=backbone_config["repo"],
                    decoder_repo=codec_config["repo"],
                    device=backbone_device,
                    backend="lmdeploy" if force_lmdeploy and "GPU" in backbone_choice else "standard",
                    hf_token=custom_hf_token
                )
            else:
                from vieneu.standard import VieNeuTTS
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token,
                    gguf_filename=backbone_config.get("gguf_filename")
                )

            # Perform LoRA Merge if needed (ONLY for Standard Backend)
            # For LMDeploy, we handled it above by saving to disk
            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                try:
                    # 1. Load Adapter
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    
                    # 2. Merge and Unload
                    # Check if backbone matches expected type for merge
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        print("   🔄 Merging LoRA into backbone...")
                        tts.backbone = tts.backbone.merge_and_unload()
                        
                        # Reset LoRA state so it behaves like a normal model
                        tts._lora_loaded = False 
                        tts._current_lora_repo = None
                        print("   ✅ Merged successfully!")
                    else:
                        print("   ⚠️ Warning: Model does not support merge_and_unload, keeping adapter active.")
                        
                except Exception as e:
                     raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ❌"
            )
        
        warning_msg = ""
        if lmdeploy_error_reason:
             warning_msg = (
                 f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                 f"👉 {lmdeploy_error_reason}\n"
                 f"💡 Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
             )

        success_msg = get_model_status_message()
        if warning_msg:
            success_msg += warning_msg
            
        # Voices the user saved from the Voice Cloning tab (v3 only), on top of the built-ins.
        try:
            _saved = load_user_voices(tts)
            if _saved:
                print(f"   💾 Loaded {len(_saved)} saved voice(s): {', '.join(_saved)}")
        except Exception as _e:  # noqa: BLE001
            print(f"   ⚠️ Không đọc được giọng đã lưu: {_e}")

        # Prepare voice update
        try:
            # Get voices with descriptions for UI from SDK
            voices = tts.list_preset_voices()
        except Exception:
            voices = []

        has_voices = len(voices) > 0
        
        if has_voices:
            default_v = tts._default_voice
            
            # Helper to get values list
            is_tuple = (len(voices) > 0 and isinstance(voices[0], tuple))
            voice_values = [v[1] for v in voices] if is_tuple else voices
            
            if not default_v and voice_values:
                 default_v = voice_values[0]

            # Ensure default_v is in the list and selected correctly
            if default_v and default_v not in voice_values:
                if is_tuple:
                    # Try to find a nice description if possible, else use ID
                    voices.append((default_v, default_v))
                else:
                    voices.append(default_v)
            
            # Editors' picks first, then A-Z
            _sort_voices(tts, voices)

            voice_update = gr.update(choices=with_switches(voices), value=default_v, interactive=True)
            
            global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
            PRESET_VOICES_CACHE = voices
            
            # Filter voices for conversation tab (podcast=True)
            # Handle both boolean True/False and string "True"/"False"
            def _check_podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get('podcast', True)
                if isinstance(val, str):
                    return val.strip().lower() == "true"
                return bool(val)

            CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1])]
            
            slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
            
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))
        else:
            # Missing voices.json case
            msg = "⚠️ Không tìm thấy file voices.json. Vui lòng dùng Tab Voice Cloning."
            voice_update = gr.update(choices=[msg], value=msg, interactive=False)
            slot_dd_update = gr.update(choices=[])
            
            tab_c = gr.update(visible=_supports_cloning(backbone_choice))

        # Conversation tab: available for v2 (sequential) and v3 Turbo (batched).
        is_v2 = (backbone_choice == "VieNeu-TTS-v2 (GPU)" or backbone_choice == "VieNeu-TTS-v2 (CPU)")
        is_v3_conv = "v3" in (backbone_choice or "").lower()
        conv_tab_update = gr.update(visible=is_v2 or is_v3_conv)

        # Update all MAX_SPEAKERS slot dropdowns
        slot_updates = [slot_dd_update] * MAX_SPEAKERS

        yield (
            success_msg,
            gr.update(interactive=True), # btn_generate
            gr.update(interactive=True), # btn_generate_conv
            gr.update(interactive=True), # btn_load
            gr.update(interactive=False), # btn_stop
            voice_update,
            tab_c,
            conv_tab_update,
            *slot_updates
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=False), # btn_generate_conv
                gr.update(interactive=True), # btn_load
                gr.update(interactive=False), # btn_stop
                gr.update(), # voice_select
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )
        else: 
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )


FT_SWITCH = "🎓→"  # value prefix of a voice-list entry that stands for a voice of another model


def model_switch_entries() -> list:
    """Voice-list entries for voices that need another model: one per fine-tuned model on this PC (every
    ``finetune/output/<voice>/merged`` found at start, so later ones appear by themselves), and, while a
    fine-tuned model is loaded, the way back to the built-in voices. Choosing one loads that model."""
    entries = []
    for label in BACKBONE_CONFIGS:
        if "🎓" in label and label != current_backbone:
            voice = label.split("🎓", 1)[1].strip()
            entries.append((f"🎓 {voice} (giọng fine-tune: chọn để tự nạp model của giọng này, 30–40 giây)", FT_SWITCH + label))
    if "🎓" in (current_backbone or ""):
        base = next(iter(BACKBONE_CONFIGS))
        entries.append((f"↩️ Các giọng có sẵn (chọn để tự nạp lại {base})", FT_SWITCH + base))
    return entries


def with_switches(voices) -> list:
    return list(voices or []) + model_switch_entries()


def resolve_voice_id(v_id: str) -> str:
    """Robustly resolve voice ID, handling both display labels and internal IDs."""
    if not v_id:
        return v_id
    
    global PRESET_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return v_id
        
    for item in PRESET_VOICES_CACHE:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, value = item[0], item[1]
            if v_id == value or v_id == label:
                return value
        else:
            if v_id == item:
                return item
            
    return v_id

# --- 2. DATA & HELPERS ---

# Phong cách đọc (style) đã bỏ trên v3 Turbo: style nằm sẵn trong reference
# (speaker embedding + ref codes) nên mọi lần sinh đều là giọng tự nhiên.


def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int,
                      denoise_ref: bool = True, session_id: str = None):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()  # Reset for new generation
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    
    # Setup Reference
    yield None, "📄 Đang xử lý Reference..."
    
    is_v3 = "v3" in (current_backbone or "").lower()
    v3_speaker_emb = None
    try:
        ref_codes = None
        ref_text_raw = ""
        v3_voice_token_id = None
        v_id = None

        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Vui lòng chọn giọng mẫu.")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning.")

            # Use SDK method - handles caching and JSON internally
            v_id = resolve_voice_id(voice_choice)
            voice_data = tts.get_preset_voice(v_id)
            if is_v3:
                # v3 presets carry a speaker embedding + reference codes.
                v3_speaker_emb = voice_data['speaker_emb']
                ref_codes = voice_data.get('codes')
            else:
                ref_codes = voice_data['codes']
                ref_text_raw = voice_data.get('text', '')
                v3_voice_token_id = voice_data.get('reserved_id')

        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")

            cb_lower = (current_backbone or "").lower()
            # Turbo v2 and v3 clone purely from audio → no reference transcript needed.
            needs_ref_text = "v2-turbo" not in cb_lower and "v3" not in cb_lower
            if needs_ref_text and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")

            ref_text_raw = custom_text.strip() if custom_text else ""
            if is_v3:
                # Enroll once: (optionally) denoise + trim ≤8s → speaker embedding + ref codes.
                v3_speaker_emb, ref_codes = tts.encode_reference(custom_audio, denoise=denoise_ref)
            else:
                ref_codes = tts.encode_reference(custom_audio)

        # Ensure numpy for inference
        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ Lỗi xử lý Reference Audio: {str(e)}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        # ============================ v3 TURBO BRANCH ========================
        # VieNeu-TTS v3 Turbo: split the text into chunks and run them through the
        # batched serving engine (vieneu.v3_turbo_serve) so multiple chunks share
        # each forward step (big GPU throughput win at Batch Size 16–64). Falls back
        # to single-utterance generation on CPU / 1 chunk / batching disabled.
        if "v3" in (current_backbone or "").lower():
            _t0 = time.time()
            # v3 Nano shares this branch: its engine.infer(phonemes=, speaker_emb=, ref_codes=)
            # has the same shape (ref_codes = the voice's style tokens) and its device is
            # always CPU, so it takes the sequential path below.
            v3_label = "v3 Nano" if "nano" in (current_backbone or "").lower() else "v3 Turbo"
            yield None, f"⏳ Đang tổng hợp ({v3_label})..."
            sr_v3 = getattr(tts, "sample_rate", 48000)
            try:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions

                # Chia chunk theo TEXT đã normalize (giống v2-gpu, không vụn), giữ
                # inline cues; phonemize TỪNG chunk khi dựng request.
                v3_chunks, v3_gaps = normalize_to_chunks_v3_with_gaps(raw_text, max_chars=max_chars_chunk)
                v3_bs = max(1, int(max_batch_size_run)) if use_batch else 1
                v3_engine_dev = getattr(getattr(tts, "engine", None), "device", None)
                v3_can_batch = (
                    v3_bs > 1 and len(v3_chunks) > 1
                    and v3_engine_dev is not None and v3_engine_dev.type == "cuda"
                )

                if v3_can_batch:
                    from vieneu.v3_turbo_serve import V3TurboBatchEngine
                    if getattr(tts, "_v3_batch_engine", None) is None:
                        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)
                    # Phonemize từng TEXT chunk (giữ inline cues) rồi gộp lô theo ĐỘ DÀI
                    # phoneme (length bucketing): các prompt cùng cỡ vào chung lô để
                    # giảm left-padding — đỡ tốn prefill và đỡ nhiễu số học do pad.
                    # Kết quả được trả về đúng vị trí gốc nên khớp v3_gaps khi join.
                    v3_phs = [phonemize_text_with_emotions(c) for c in v3_chunks]
                    v3_order = sorted(range(len(v3_chunks)), key=lambda k: len(v3_phs[k]))
                    v3_wavs = [None] * len(v3_chunks)
                    for bi, i in enumerate(range(0, len(v3_order), v3_bs)):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        idxs = v3_order[i:i + v3_bs]
                        yield None, f"⚡ v3 Turbo: lô {bi + 1} ({len(idxs)} đoạn, batch size {v3_bs})..."
                        reqs = [{"phonemes": v3_phs[j], "speaker_emb": v3_speaker_emb,
                                 "ref_codes": ref_codes, "use_ref_codes": True} for j in idxs]
                        for j, w in zip(idxs, tts._v3_batch_engine.generate_batch(
                                reqs, temperature=temperature, max_new_frames=300)):
                            v3_wavs[j] = w
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
                else:
                    # CPU (ONNX) hoặc GPU khi tắt batch: xử lý TUẦN TỰ từng đoạn.
                    # Gọi engine 1 lần / đoạn (giống nhánh batch nhưng tuần tự) để có
                    # ĐÚNG 1 wav / đoạn -> khớp với v3_gaps khi join, đồng thời báo cho
                    # người dùng đang xử lý đến đoạn nào + ước tính thời gian còn lại —
                    # quan trọng trên CPU vì mỗi đoạn có thể mất nhiều giây.
                    # (KHÔNG enumerate tts.infer(): infer() trả về 1 mảng đã ghép sẵn,
                    #  lặp qua nó sẽ ra từng sample numpy.float32 -> len() ném lỗi.)
                    total_v3 = len(v3_chunks)
                    v3_wavs = []
                    chunk_durations = []
                    last_t = time.time()
                    for i, chunk in enumerate(v3_chunks):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        yield None, f"⏳ {v3_label}: Đang xử lý đoạn {i + 1}/{total_v3}..."
                        ph = phonemize_text_with_emotions(chunk)
                        chunk_wav = tts.engine.infer(
                            phonemes=ph, speaker_emb=v3_speaker_emb, ref_codes=ref_codes,
                            use_ref_codes=True,
                            temperature=temperature, max_new_frames=300)
                        now = time.time()
                        chunk_durations.append(now - last_t)
                        last_t = now
                        # Luôn append (kể cả rỗng) để v3_wavs khớp 1-1 với v3_chunks —
                        # lọc bớt ở đây sẽ làm lệch v3_gaps khi join.
                        v3_wavs.append(chunk_wav if chunk_wav is not None
                                       else np.zeros(0, dtype=np.float32))
                        done = i + 1
                        if done < total_v3:
                            avg = sum(chunk_durations) / len(chunk_durations)
                            eta = avg * (total_v3 - done)
                            yield None, (
                                f"⏳ {v3_label}: Đã xong {done}/{total_v3} đoạn "
                                f"(ước tính còn lại: {_format_duration(eta)})... "
                                f"đang xử lý đoạn {done + 1}/{total_v3}"
                            )
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
            except Exception as e:
                yield None, f"❌ Lỗi tổng hợp ({v3_label}): {str(e)}"
                return
            if wav is None or len(wav) == 0:
                yield None, "❌ Không sinh được audio nào."
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, wav, sr_v3)
                out_path_v3 = tmp.name
            _dt = time.time() - _t0
            _spd = f", Tốc độ: {len(wav)/sr_v3/_dt:.2f}x realtime" if _dt > 0 else ""
            yield out_path_v3, f"✅ Hoàn tất! ({v3_label}, Thời gian: {_dt:.2f}s{_spd})"
            cleanup_gpu_memory()
            return
        # ========================== end v3 TURBO BRANCH ======================

        backend_name = "LMDeploy" if using_lmdeploy else "Standard"

        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
            
        total_chunks = len(text_chunks)

        batch_info = " (Batch Mode)" if use_batch and using_lmdeploy and total_chunks > 1 else ""
        
        # Show batch size info
        batch_size_info = ""
        if use_batch and using_lmdeploy and hasattr(tts, 'max_batch_size'):
            batch_size_info = f" [Max batch: {tts.max_batch_size}]"
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."
        
        all_wavs = []
        sr = 24000
        
        start_time = time.time()
        
        try:
            if is_v2_turbo:
                # Sequential processing with progress updates
                total_chunks = len(text_chunks)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."
                    
                    chunk_wav = tts.infer(
                        chunk.text, 
                        ref_codes=ref_codes, 
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True,
                        skip_phonemize=True
                    )
                    
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
                        # Add silence between Gradio-level chunks for Turbo
                        if i < total_chunks - 1:
                            sil_dur = get_silence_duration_v2(chunk)
                            sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                            all_wavs.append(sil_wav)
            
            # Use batch processing if enabled and using LMDeploy (for v1)
            elif use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Process in mini-batches to allow cancellation between batches
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                total_batch_duration = 0.0
                completed_batches = 0
                
                for i in range(0, total_chunks, max_batch_size_run):
                    if _STOP_EVENT.is_set():
                        print("🛑 Synthesis stopped during batch processing.")
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    
                    batch_idx = i // max_batch_size_run
                    estimate_info = ""
                    if completed_batches > 0:
                        average_batch_duration = total_batch_duration / completed_batches
                        estimated_total = average_batch_duration * num_batches
                        estimated_remaining = average_batch_duration * max(0, num_batches - batch_idx)
                        estimate_info = (
                            f" | Ước tính còn lại: {_format_duration(estimated_remaining)}"
                            f" / tổng: {_format_duration(estimated_total)}"
                        )
                    yield None, f"⚡ Đang xử lý batch {batch_idx+1}/{num_batches} (đoạn {i+1}-{min(i+max_batch_size_run, total_chunks)}){estimate_info}..."
                    
                    current_batch = text_chunks[i : i + max_batch_size_run]
                    batch_start_time = time.time()
                    batch_wavs = tts.infer_batch(
                        current_batch, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        max_batch_size=max_batch_size_run,
                        temperature=temperature,
                        skip_normalize=True
                    )
                    batch_duration = time.time() - batch_start_time
                    total_batch_duration += batch_duration
                    completed_batches += 1
                    average_batch_duration = total_batch_duration / completed_batches
                    estimated_total = average_batch_duration * num_batches
                    estimated_remaining = average_batch_duration * max(0, num_batches - completed_batches)
                    for chunk_wav in batch_wavs:
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
                    yield None, (
                        f"✅ Xong batch {batch_idx+1}/{num_batches} "
                        f"(trung bình batch: {_format_duration(average_batch_duration)}, "
                        f"ước tính còn lại: {_format_duration(estimated_remaining)}, "
                        f"tổng: {_format_duration(estimated_total)})"
                    )

            else:
                # Sequential processing (PyTorch or GGUF v1)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wav = tts.infer(
                        chunk, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            # Use utility function for joining with silence/crossfade
            # Default silence=0.15s to match SDK
            silence_p = 0.15 if not is_v2_turbo else 0.0 # Turbo adds silence internally
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()
            
        except Exception as e:
            # Check for CUDA OOM specifically if torch is loaded
            if 'torch' in sys.modules:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    cleanup_gpu_memory()
                    yield None, (
                        f"❌ GPU hết VRAM! Hãy thử:\n"
                        f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                        f"• Giảm độ dài văn bản\n\n"
                        f"Chi tiết: {str(e)}"
                    )
                    return
            
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        break
                    
                    if is_v2_turbo:
                        stream_gen = tts.infer_stream(
                            chunk_text.text,
                            ref_codes=ref_codes, 
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            skip_phonemize=True,
                            emotion_tag=""
                        )
                    else:
                        stream_gen = tts.infer_stream(
                            chunk_text, 
                            ref_codes=ref_codes, 
                            ref_text=ref_text_raw,
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            emotion_tag=""
                        )
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if _STOP_EVENT.is_set():
                            break
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                            
                    # Add silence between chunks for Turbo v2
                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()

DEFAULT_CLONE_TEXT = "Xin chào, đây là giọng nói vừa được nhân bản từ đoạn audio mẫu của bạn. Nghe có giống không?"

synthesize_speech_with_estimate = wrap_with_estimate(synthesize_speech)

def synthesize_conversation_with_empty_estimate(*args):
    for audio_path, status in synthesize_conversation(*args):
        yield audio_path, status, ""

# --- CANCELLATION ---
# threading.Event is a mutable object: never reassigned, always the same reference.
# All threads share the exact same object — no scoping/serialization issues.
_STOP_EVENT = threading.Event()

# --- 3. CONVERSATION LOGIC ---

def _synthesize_conversation_v3(lines, mapping, temperature, max_chars_chunk, silence_duration):
    """v3 Turbo conversation: batch the WHOLE conversation at batch size 32.

    Unlike the v2 path (sequential, one turn at a time), every turn is split into
    chunks and ALL chunks across ALL turns are flattened into one request list, then
    fed to the V3 batched engine in groups of 32 — regardless of which speaker each
    chunk belongs to (each request carries its own speaker ref_codes + reserved
    token). Wavs are reassembled per turn (in order) and joined with inter-turn
    silence. Yields ``(audio_path, status)`` like the v2 path. 48 kHz.
    """
    global tts
    from collections import defaultdict
    from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions, normalize_to_chunks_v3_with_gaps
    # NOTE: KHÔNG import vieneu.v3_turbo_serve ở đây — module đó import torch ở cấp
    # module, nên trên bản cài CPU/macOS không-torch (ONNX) sẽ lỗi "No module named
    # 'torch'". Chỉ import bên trong nhánh CUDA bên dưới (nơi thực sự cần batch engine).

    sr = getattr(tts, "sample_rate", 48000)
    t0 = time.time()

    # Resolve each speaker → (speaker_emb, ref_codes), cached per speaker.
    def _voice_for(spk_name):
        cfg = mapping.get(spk_name.lower())
        v_id = (cfg or {}).get('voice') or tts._default_voice
        try:
            vd = tts.get_preset_voice(v_id)
        except Exception:
            vd = tts.get_preset_voice(tts._default_voice)
        emb = vd.get('speaker_emb')
        rc = vd.get('codes')
        if 'torch' in sys.modules:
            import torch
            if isinstance(rc, torch.Tensor):
                rc = rc.cpu().numpy()
        return (np.asarray(emb, dtype=np.float32) if emb is not None else None,
                np.asarray(rc) if rc is not None else None)

    # CPU (ONNX) has no batched engine → run sequentially, one turn at a time.
    dev = getattr(getattr(tts, "engine", None), "device", None)
    is_cuda = dev is not None and getattr(dev, "type", None) == "cuda"
    if not is_cuda:
        all_wavs = []
        for li, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            cfg = mapping.get(line['speaker'].lower())
            v_id = (cfg or {}).get('voice') or tts._default_voice
            yield None, f"⏳ [{li+1}/{len(lines)}] {line['speaker']}: {line['text'][:30]}..."
            try:
                wav = tts.infer(line['text'], voice=v_id,
                                temperature=temperature, max_chars=max_chars_chunk)
            except Exception as e:
                print(f"❌ Lỗi câu {li+1}: {e}")
                continue
            if wav is not None and len(wav):
                all_wavs.append(wav)
                if li < len(lines) - 1 and silence_duration > 0:
                    all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))
        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {time.time()-t0:.1f}s, CPU tuần tự)"
        cleanup_gpu_memory()
        return

    voice_cache = {}
    reqs, req_line = [], []
    line_gaps = {}
    for li, line in enumerate(lines):
        key = line['speaker'].lower()
        if key not in voice_cache:
            voice_cache[key] = _voice_for(line['speaker'])
        spk_emb, ref_codes = voice_cache[key]
        # Chia chunk theo TEXT đã normalize (giống v2-gpu), giữ inline cues, rồi
        # phonemize từng chunk. Hội thoại luôn dùng phong cách Tự nhiên.
        line_chunks, line_gaps[li] = normalize_to_chunks_v3_with_gaps(line['text'], max_chars=max_chars_chunk)
        for chunk in line_chunks:
            reqs.append({"phonemes": phonemize_text_with_emotions(chunk),
                         "speaker_emb": spk_emb, "ref_codes": ref_codes,
                         "use_ref_codes": True})
            req_line.append(li)

    if not reqs:
        yield None, "❌ Không có lời thoại để tổng hợp."
        return

    if getattr(tts, "_v3_batch_engine", None) is None:
        # Chỉ tới đây khi chạy trên CUDA → torch chắc chắn có sẵn.
        from vieneu.v3_turbo_serve import V3TurboBatchEngine
        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)

    BS = 32
    total_batches = (len(reqs) + BS - 1) // BS
    # Length bucketing: gộp lô theo độ dài phoneme để giảm left-padding; kết quả
    # đặt về đúng vị trí gốc nên vẫn khớp req_line khi gom theo lời thoại.
    req_order = sorted(range(len(reqs)), key=lambda k: len(reqs[k]["phonemes"]))
    wavs_flat = [None] * len(reqs)
    for bi, i in enumerate(range(0, len(req_order), BS)):
        if _STOP_EVENT.is_set():
            yield None, "⏹️ Đã dừng hội thoại."
            return
        idxs = req_order[i:i + BS]
        yield None, f"⚡ v3 Turbo hội thoại: lô {bi + 1}/{total_batches} ({len(idxs)} đoạn, batch 32)..."
        for j, w in zip(idxs, tts._v3_batch_engine.generate_batch(
                [reqs[k] for k in idxs], temperature=temperature, max_new_frames=300)):
            wavs_flat[j] = w

    # Reassemble per turn (in order), then join turns with inter-turn silence.
    by_line = defaultdict(list)
    for w, li in zip(wavs_flat, req_line):
        by_line[li].append(w)

    all_wavs = []
    for li in range(len(lines)):
        lw = join_audio_chunks(by_line[li], sr=sr, silence_ps=gaps_to_silence(line_gaps.get(li, []))) if by_line[li] else None
        if lw is None or len(lw) == 0:
            continue
        all_wavs.append(lw)
        if li < len(lines) - 1 and silence_duration > 0:
            all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))

    if not all_wavs:
        yield None, "❌ Không thể tạo được âm thanh nào!"
        return

    yield None, "🪄 Đang ghép nối âm thanh..."
    final_wav = np.concatenate(all_wavs)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, final_wav, sr)
        elapsed = time.time() - t0
        yield tmp.name, f"✅ Hoàn tất hội thoại! ({len(lines)} câu, {len(reqs)} đoạn, {elapsed:.1f}s, batch 32)"
    cleanup_gpu_memory()


def synthesize_conversation(
    script_text: str,
    *args
):
    """
    Synthesizes multi-speaker conversation from a script.

    Gradio passes speaker name boxes and voice dropdowns as individual positional args.
    Layout: args[0..MAX_SPEAKERS-1] = speaker names, args[MAX_SPEAKERS..2*MAX_SPEAKERS-1] = voice IDs,
    args[2*MAX_SPEAKERS] = silence_duration, args[2*MAX_SPEAKERS+1] = temperature,
    args[2*MAX_SPEAKERS+2] = max_chars_chunk, args[2*MAX_SPEAKERS+3] = session_id
    """
    speaker_names     = list(args[:MAX_SPEAKERS])
    speaker_voices    = list(args[MAX_SPEAKERS:MAX_SPEAKERS*2])
    silence_duration  = args[MAX_SPEAKERS * 2]
    temperature       = args[MAX_SPEAKERS * 2 + 1]
    max_chars_chunk   = args[MAX_SPEAKERS * 2 + 2]
    session_id        = args[MAX_SPEAKERS * 2 + 3] if len(args) > MAX_SPEAKERS * 2 + 3 else None

    global tts, model_loaded, using_lmdeploy
    
    _STOP_EVENT.clear()
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
        
    if not script_text or script_text.strip() == "":
        yield None, "⚠️ Vui lòng nhập kịch bản hội thoại!"
        return

    # 1. Parse Script
    lines = []
    for line in script_text.strip().split('\n'):
        if not line.strip(): continue
        if ':' in line:
            parts = line.split(':', 1)
            lines.append({'speaker': parts[0].strip(), 'text': parts[1].strip()})
        else:
            if lines:
                lines[-1]['text'] += " " + line.strip()
            else:
                lines.append({'speaker': 'Narrator', 'text': line.strip()})

    if not lines:
        yield None, "⚠️ Không tìm thấy lời thoại hợp lệ (định dạng Nhân vật: Lời thoại)!"
        return

    # 2. Build Speaker Mapping from individual slot components
    mapping = {}
    for name, voice in zip(speaker_names, speaker_voices):
        name = str(name).strip() if name else ""
        if not name: continue
        # Use lowercase key for robust matching
        v_id = resolve_voice_id(str(voice)) if voice else ""
        mapping[name.lower()] = {
            'type': 'Preset',
            'voice': v_id,
            'ref_text': ''
        }

    # 2b. v3 Turbo: batch toàn bộ hội thoại (bs=32), bất kể speaker thay đổi.
    if "v3" in (current_backbone or "").lower():
        yield from _synthesize_conversation_v3(
            lines, mapping, temperature, max_chars_chunk, silence_duration
        )
        return

    # 3. Process Each Line
    all_wavs = []
    sr = 24000
    total_lines = len(lines)
    
    yield None, f"🎭 Đang khởi tạo hội thoại ({total_lines} câu)..."
    
    start_time = time.time()
    
    try:
        for i, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            spk_name = line['speaker']
            text = line['text']
            
            yield None, f"⏳ [{i+1}/{total_lines}] {spk_name}: {text[:30]}..."
            
            # Determine voice
            ref_codes = None
            ref_text_val = None
            current_voice_obj = None
            
            # Case-insensitive lookup
            config = mapping.get(spk_name.lower())
            
            if not config:
                print(f"  ⚠️ Character '{spk_name}' not found in mapping. Fallback to default.")
                # Fallback to default if speaker not mapped
                try:
                    # Get default voice data
                    default_v_id = tts._default_voice
                    if not default_v_id:
                        dv_list = tts.list_preset_voices()
                        if dv_list:
                            first = dv_list[0]
                            default_v_id = first[1] if isinstance(first, tuple) else first
                    
                    if default_v_id:
                        current_voice_obj = tts.get_preset_voice(default_v_id)
                        ref_codes = current_voice_obj['codes']
                        ref_text_val = current_voice_obj['text']
                except Exception as e:
                    print(f"  ❌ Fallback failed: {e}")
            else:
                try:
                    v_id = config['voice']
                    if config['type'] == "Preset":
                        current_voice_obj = tts.get_preset_voice(v_id)
                        if current_voice_obj and 'codes' in current_voice_obj:
                            ref_codes = current_voice_obj['codes']
                            ref_text_val = current_voice_obj['text']
                        else:
                            print(f"  ❌ Could not find codes for voice '{v_id}'")
                    else: # Custom
                        if v_id and os.path.exists(v_id):
                            ref_codes = tts.encode_reference(v_id)
                            ref_text_val = config.get('ref_text', '')
                            current_voice_obj = {'codes': ref_codes, 'text': ref_text_val}
                            print(f"  🦜 Using custom voice for '{spk_name}'")
                except Exception as e:
                    print(f"  ❌ Lỗi nạp giọng cho {spk_name} (ID: {config.get('voice')}): {e}")
            
            # Ensure numpy for inference
            if 'torch' in sys.modules:
                import torch
                if isinstance(ref_codes, torch.Tensor):
                    ref_codes = ref_codes.cpu().numpy()

            # Infer audio
            try:
                wav = tts.infer(
                    text,
                    voice=current_voice_obj, # Use full voice object
                    ref_codes=ref_codes,     # Fallback if object not supported
                    ref_text=ref_text_val,
                    temperature=temperature,
                    max_chars=max_chars_chunk,
                    emotion_tag="<|emotion_0|>" # Emotion tag for conversation
                )
                
                all_wavs.append(wav)
                
                # Add silence between turns
                if i < total_lines - 1 and silence_duration > 0:
                    silence_len = int(sr * silence_duration)
                    silence = np.zeros(silence_len)
                    all_wavs.append(silence)
                    
            except Exception as e:
                print(f"❌ Lỗi tổng hợp câu {i+1}: {e}")
                continue

        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return

        # 4. Merge and Output
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, sr)
            elapsed = time.time() - start_time
            yield tmp.name, f"✅ Hoàn tất hội thoại! ({total_lines} câu, xử lý trong {elapsed:.1f}s)"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ Lỗi hệ thống: {str(e)}"

def extract_speakers_from_script(script):
    """Find unique speakers and return gr.update() lists for the 8 slot components."""
    global CONV_VOICES_CACHE
    if not script:
        # Hide all slots
        name_updates = [gr.update(value="", visible=False)] * MAX_SPEAKERS
        dd_updates   = [gr.update(value=None, visible=False)] * MAX_SPEAKERS
        row_updates  = [gr.update(visible=False)] * MAX_SPEAKERS
        return name_updates + dd_updates + row_updates

    speakers = []
    seen = set()
    for line in script.strip().split('\n'):
        if ':' in line:
            s = line.split(':', 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                speakers.append(s)

    # Auto-match each speaker name to a preset voice
    def _best_match(name):
        if not CONV_VOICES_CACHE:
            return None
        
        name_l = name.lower()
        
        # 0. Manual overrides for specific common names
        overrides = {
            "phương": "Trúc Ly",
            "dũng": "Thanh Bình",
            "hùng": "Thái Sơn"
        }
        if name_l in overrides:
            target = overrides[name_l].lower()
            for v in CONV_VOICES_CACHE:
                label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
                if target in label.lower() or target in value.lower():
                    return value

        # 1. Try to find name in labels or values
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l == label.lower() or name_l == value.lower():
                return value
        
        # 2. Fuzzy match (contains)
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l in label.lower() or name_l in value.lower() or label.lower() in name_l or value.lower() in name_l:
                return value
        
        # 3. Default to first voice if no match
        first_voice = CONV_VOICES_CACHE[0]
        return first_voice[1] if isinstance(first_voice, tuple) else first_voice

    name_updates, dd_updates, row_updates = [], [], []
    for i in range(MAX_SPEAKERS):
        if i < len(speakers):
            name_updates.append(gr.update(value=speakers[i], visible=True))
            dd_updates.append(gr.update(value=_best_match(speakers[i]), choices=CONV_VOICES_CACHE, visible=True))
            row_updates.append(gr.update(visible=True))
        else:
            name_updates.append(gr.update(value="", visible=False))
            dd_updates.append(gr.update(value=None, choices=CONV_VOICES_CACHE, visible=False))
            row_updates.append(gr.update(visible=False))

    return name_updates + dd_updates + row_updates

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF (fitz)."""
    global HAS_FITZ, fitz
    if not HAS_FITZ:
        return "⚠️ PyMuPDF chưa được cài đặt. Vui lòng chạy: pip install PyMuPDF"
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages_text.append(text.strip())
        doc.close()
        if not pages_text:
            return "⚠️ Không tìm thấy văn bản trong file PDF."
        full_text = "\n\n".join(pages_text)
        return full_text
    except Exception as e:
        return f"⚠️ Lỗi khi đọc PDF: {str(e)}"

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]

# Gradio asks the browser for the microphone with its defaults, and the browser then filters the sound for
# calls: noise suppression mutes soft parts of a sentence and the automatic gain changes its loudness. Voice
# recordings for cloning and fine-tuning need the microphone as it is.
RAW_MIC_HEAD = """
<script>
(() => {
  const devices = navigator.mediaDevices;
  if (!devices || !devices.getUserMedia) return;
  const ask = devices.getUserMedia.bind(devices);
  devices.getUserMedia = (constraints) => {
    if (constraints && constraints.audio) {
      const audio = constraints.audio === true ? {} : constraints.audio;
      constraints = {...constraints, audio: {...audio, echoCancellation: false, noiseSuppression: false, autoGainControl: false}};
    }
    return ask(constraints);
  };
})();
</script>
"""

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS", head=head_html + RAW_MIC_HEAD) as demo:
    # Session ID for cancellation tracking
    session_id_state = gr.State("")

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio</span>
    </h1>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>Models:</strong>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
            <span>•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2" target="_blank" class="model-card-link">VieNeu-TTS-v2</a>
            <span>•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo" target="_blank" class="model-card-link">VieNeu-TTS-v3-Turbo</a>
        </div>
        <div class="model-card-item">
            <strong>Repository:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
        </div>
        <div class="model-card-item">
            <strong>Tác giả:</strong>
            <a href="https://www.facebook.com/pnnbao97" target="_blank" class="model-card-link">Phạm Nguyễn Ngọc Bảo</a>
        </div>
        <div class="model-card-item">
            <strong>Discord:</strong>
            <a href="https://discord.gg/yJt8kzjzWZ" target="_blank" class="model-card-link">Tham gia cộng đồng</a>
        </div>
    </div>
</div>
        """)
        
        # --- CONFIGURATION ---
        with gr.Group():
            with gr.Row():
                # --- BACKBONE & CODEC DEFAULT LOGIC ---
                # "VieNeu-TTS-v3-Turbo" (fp32) is the first entry on both CPU and GPU, so it is
                # the default everywhere. The int8 Turbo build (CPU only) and v3 Nano (preview)
                # are opt-in choices listed after it.
                default_backbone = list(BACKBONE_CONFIGS.keys())[0]
                
                # Default parameters based on backbone
                if "v3" in default_backbone.lower():
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.8
                    default_text = DEFAULT_TEXT_V3
                elif "Turbo" in default_backbone:
                    default_codec = "VieNeu-Codec"
                    default_temp = 0.4
                    default_text = DEFAULT_TEXT_TURBO
                elif "(CPU)" in default_backbone:
                    default_codec = "NeuCodec (ONNX)"
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU
                else:
                    default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                    default_temp = 0.7
                    default_text = DEFAULT_TEXT_GPU

                # v3 Turbo batches chunks through the serving engine → default 32.
                # Must be set at creation: v3 is the default backbone, so the
                # on_backbone_change handler (which also sets 32) never fires on load.
                # 16 is safe on 8 GB cards; the slider goes to 64 for bigger GPUs.
                default_batch_size = 16 if "v3" in default_backbone.lower() else 4

                backbone_select = gr.Dropdown(
                    list(BACKBONE_CONFIGS.keys()) + ["Custom Model"],
                    value=default_backbone,
                    label="🦜 Backbone"
                )
                codec_select = gr.Dropdown(
                    list(CODEC_CONFIGS.keys()), 
                    value=default_codec, 
                    label="🎵 Codec",
                    interactive=False
                )
                device_choice = gr.Radio(get_available_devices(), value="Auto", label="🖥️ Device")
            
            with gr.Row(visible=False) as custom_model_group:
                custom_backbone_model_id = gr.Textbox(
                    label="📦 Custom Model ID",
                    placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                    info="Nhập HuggingFace Repo ID hoặc đường dẫn local",
                    scale=2
                )
                custom_backbone_hf_token = gr.Textbox(
                    label="🔑 HF Token (nếu private)",
                    placeholder="Để trống nếu repo public",
                    type="password",
                    info="Token để truy cập repo private",
                    scale=1
                )
                base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                custom_backbone_base_model = gr.Dropdown(
                    base_model_choices,
                    label="🔗 Base Model (cho LoRA)",
                    value=base_model_choices[0] if base_model_choices else None,
                    visible=False,
                    info="Model gốc để merge với LoRA (GPU Only)",
                    scale=1
                )
            
            with gr.Row():
                use_lmdeploy_cb = gr.Checkbox(
                    value=True,
                    label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                    info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể.",
                    visible="v3" not in default_backbone.lower(),  # v3 Turbo (PyTorch) không dùng LMDeploy
                )
            
            
            gr.Markdown("""
            💡 **Sử dụng Custom Model:** Chọn "Custom Model" để tải LoRA adapter hoặc bất kỳ model nào được finetune từ **VieNeu-TTS** hoặc **VieNeu-TTS-0.3B**.
            """)
            
            gr.HTML("""
            <div class="warning-banner">
                <div class="warning-banner-title">
                    🦜 Gợi ý tối ưu hiệu năng
                </div>
                <div class="warning-banner-grid">
                    <div class="warning-banner-item">
                        <strong>🐆 Hệ máy GPU</strong>
                        <div class="warning-banner-content">
                            <b>VieNeu-TTS-v3-Turbo</b> — 48kHz, giọng mặc định ổn định, Voice Cloning và hỗ trợ các tag cảm xúc `[cười]` `[hắng giọng]` `[thở dài]` (riêng tag cảm xúc vẫn đang thử nghiệm). Nếu gặp lỗi hãy báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ.
                        </div>
                    </div>
                    <div class="warning-banner-item" style="background: #dcfce7; border-color: #86efac;">
                        <strong style="color: #15803d;">🐢 Hệ máy CPU</strong>
                        <div class="warning-banner-content" style="color: #166534;">
                            Máy <b>CPU</b> dùng bản mặc định <b>VieNeu-TTS-v3-Turbo</b> (ONNX, chất lượng tối đa). Bản <b>VieNeu-TTS-v3-Nano (preview)</b> chạy rất nhanh, dành cho edge device như Android hoặc CPU yếu, nhưng <b>chất lượng kém hơn nhiều</b> so với Turbo (nhất là tiếng Anh, song ngữ) — chỉ dùng khi thực sự cần tốc độ hoặc deploy trên điện thoại. Nano đang trong quá trình thử nghiệm nên còn nhiều thiếu sót.
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.85rem; color: #92400e; border-top: 1px dashed #fcd34d; padding-top: 8px;">
                    💡 <b>Mẹo:</b> Nếu máy bạn có GPU mà không thấy các phiên bản GPU hãy xem lại cách cài đặt uv sync --group gpu
                </div>
            </div>
            """)

            gr.Markdown(
                "🆕 **VieNeu-TTS-v3-Turbo** đã **phát hành chính thức** — "
                "48kHz, giọng mặc định ổn định, **hỗ trợ Voice Cloning** (tính năng clone chỉ có từ **v3** trở lên; v1/v2 không hỗ trợ).\n\n"
                "🎭 v3 Turbo **hỗ trợ các tag cảm xúc** `[cười]` `[hắng giọng]` `[thở dài]` "
                "(chèn trực tiếp vào văn bản) — riêng tính năng này vẫn **đang thử nghiệm**."
            )

            btn_load = gr.Button("🔄 Nạp model vào bộ nhớ", variant="primary")
            model_status = gr.Markdown("⏳ Chưa nạp model. File model chỉ tải về một lần và được lưu trên máy; "
                                       "mỗi lần mở app chỉ cần nạp lại vào bộ nhớ (tự động khi chạy bằng run_windows.bat).")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                with gr.Tabs() as main_input_tabs:
                    # --- TAB 1: SINGLE SPEAKER ---
                    with gr.Tab("🦜 Đọc truyện", id="single_tab") as single_tab:
                        with gr.Accordion("📄 Tải lên PDF để trích xuất văn bản", open=False):
                            gr.Markdown(
                                "Tải lên file PDF, văn bản sẽ được tự động trích xuất và điền vào ô bên dưới. "
                                "Bạn có thể chỉnh sửa lại trước khi tạo audio."
                            )
                            with gr.Row():
                                pdf_upload = gr.File(
                                    label="📄 Chọn file PDF",
                                    file_types=[".pdf"],
                                    file_count="single",
                                    type="filepath",
                                    scale=3,
                                )
                                btn_extract_pdf = gr.Button("📄 Trích xuất văn bản", variant="secondary", scale=1, min_width=150)
                            pdf_status = gr.Markdown(visible=False)
                        text_input = gr.Textbox(
                            label="Văn bản",
                            lines=8,
                            placeholder="Dán hoặc gõ văn bản cần đọc vào đây…",
                        )
                        
                        voice_select = gr.Dropdown(
                            choices=[], value=None, label="Giọng mẫu", allow_custom_value=True,
                            info="Giọng bạn lưu ở tab Voice Cloning cũng nằm trong danh sách này.",
                        )
                        with gr.Row():
                            btn_voice_sample = gr.Button("🔊 Nghe thử giọng này", size="sm", scale=1)
                            voice_sample_audio = gr.Audio(label="Nghe thử", autoplay=True, interactive=False, scale=3)
                        generation_mode = gr.Radio(
                            ["Standard (Một lần)"],
                            value="Standard (Một lần)",
                            label="Chế độ sinh"
                        )
                        btn_generate = gr.Button("🎵 Bắt đầu", variant="primary", scale=2, interactive=False)

                    # --- TAB 2: MULTI-SPEAKER CONVERSATION ---
                    with gr.Tab("🎭 Hội thoại", id="conv_tab", visible=False) as conv_tab:
                        conv_script_input = gr.Textbox(
                            label="Kịch bản hội thoại",
                            placeholder="Phương: Chào mọi người, mình là Phương...",
                            lines=10,
                            elem_classes="script-box",
                        )
                        
                        with gr.Row():
                            btn_detect_speakers = gr.Button("🔍 Quét nhân vật", size="sm", variant="secondary")
                            silence_slider = gr.Slider(minimum=0, maximum=3, value=0.3, step=0.1, label="⏱️ Khoảng lặng (giây)")

                        gr.Markdown("### 🎭 Cấu hình giọng đọc")
                        gr.Markdown("*Nhấn **Quét nhân vật** để tự động phát hiện và ánh xạ giọng đọc. Tải model trước để có danh sách giọng.*")

                        # Pre-build MAX_SPEAKERS speaker slot rows
                        speaker_name_boxes = []
                        speaker_voice_dds  = []
                        speaker_slot_rows  = []

                        for _i in range(MAX_SPEAKERS):
                            # Mặc định cho 3 nhân vật đầu tiên theo yêu cầu
                            _default_name = ""
                            _default_voice = None
                            _row_visible = False
                            
                            if _i == 0:
                                _default_name = "Phương"
                                _default_voice = "Ly"
                                _row_visible = True
                            elif _i == 1:
                                _default_name = "Dũng"
                                _default_voice = "Binh"
                                _row_visible = True
                            elif _i == 2:
                                _default_name = "Hùng"
                                _default_voice = "Sơn"
                                _row_visible = True
                            elif _i < 2:
                                _default_name = f"Nhân vật {_i+1}"
                                _row_visible = True

                            with gr.Row(visible=_row_visible) as _row:
                                _name = gr.Textbox(
                                    value=_default_name,
                                    label="👤 Nhân vật",
                                    interactive=False,
                                    scale=1,
                                    min_width=120
                                )
                                _dd = gr.Dropdown(
                                    choices=PRESET_VOICES_CACHE,
                                    value=_default_voice,
                                    label="🎤 Giọng đọc",
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                            speaker_slot_rows.append(_row)
                            speaker_name_boxes.append(_name)
                            speaker_voice_dds.append(_dd)
                        
                        btn_generate_conv = gr.Button("🎭 Bắt đầu hội thoại", variant="primary", interactive=False)

                    # --- TAB 3: SRT → SPEECH (Vietnamese subtitles in, one audio out) ---
                    with gr.Tab("📝 SRT", id="srt_tab") as srt_tab:
                        gr.Markdown(
                            "Tải lên file **.srt tiếng Việt** (đã có lời và mốc thời gian): mỗi câu được đọc bằng "
                            "một giọng mẫu và ghép thành **một file audio** theo đúng mốc thời gian. Không dịch, không "
                            "ghép video — cần các thứ đó thì dùng <a href=\"https://www.vieneu.io/#/download\" target=\"_blank\">app VieNeu</a>. "
                            "Chỉ hỗ trợ VieNeu v3 (Turbo / Nano)."
                        )
                        srt_file = gr.File(label="📝 File phụ đề .srt", file_types=[".srt"], file_count="single", type="filepath")
                        srt_voice = gr.Dropdown(choices=[], value=None, label="Giọng mẫu", allow_custom_value=True)
                        with gr.Row():
                            srt_keep_timing = gr.Checkbox(
                                value=True, label="Giữ đúng mốc thời gian",
                                info="Chèn khoảng lặng theo phụ đề; câu nào đọc dài hơn khung thì câu sau lùi lại, không đè lên nhau. Bỏ chọn để nối liền các câu.",
                            )
                            srt_format = gr.Radio(["wav", "mp3"], value="wav", label="Định dạng xuất")
                        btn_generate_srt = gr.Button("🎵 Tạo audio từ SRT", variant="primary", interactive=False)
                        # Shipped sample so the expected .srt format is clear; clicking it
                        # loads the file into the uploader.
                        _srt_example = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "srt", "sample_vi.srt")
                        gr.Examples(
                            examples=[[_srt_example]],
                            inputs=[srt_file],
                            label="Ví dụ file .srt (bấm để nạp thử — số thứ tự, mốc thời gian, lời thoại)",
                        )

                    # --- TAB: BOOK → AUDIO (one file per chapter / per part) ---
                    with gr.Tab("📚 Sách → audio", id="book_tab") as book_tab:
                        gr.Markdown(
                            "Chọn **một file .epub** (mỗi chương thành một file audio), **một file .txt**, hoặc "
                            "**một thư mục** chứa các phần .txt / .epub (mỗi phần một file, xếp theo tên: Phần 2 trước Phần 10). "
                            "File đã xong sẽ được bỏ qua, nên bấm lại là chạy tiếp chỗ đã dừng."
                        )
                        book_source = gr.Textbox(label="📂 Đường dẫn file .epub / .txt hoặc thư mục",
                                                 placeholder=r"Ví dụ: D:\Sach\Truyen.epub hoặc D:\Sach\Truyen")
                        book_upload = gr.File(label="… hoặc kéo thả file .epub / .txt vào đây", file_types=[".epub", ".txt"],
                                              file_count="single", type="filepath")
                        book_out = gr.Textbox(label="📁 Thư mục lưu audio",
                                              placeholder="Để trống: tạo thư mục '<tên sách> - audio' cạnh file nguồn")
                        book_voice = gr.Dropdown(choices=[], value=None, label="Giọng đọc", allow_custom_value=True,
                                                 info="Giọng mẫu hoặc giọng bạn đã lưu ở tab Voice Cloning.")
                        with gr.Row():
                            btn_book_sample = gr.Button("🔊 Nghe thử giọng này", size="sm", scale=1)
                            book_sample_audio = gr.Audio(label="Nghe thử", autoplay=True, interactive=False, scale=3)
                        book_format = gr.Radio(["mp3", "opus", "wav", "flac"], value="mp3", label="Định dạng",
                                               info="opus: file nhỏ, nên dùng khi chép sang app Android. Cạnh mỗi file audio có một file .sync.json "
                                                    "(văn bản kèm mốc thời gian); chép cả hai sang điện thoại để app vừa phát vừa dò theo chữ.")
                        book_chapters = gr.Number(value=1, minimum=1, maximum=1000, precision=0, label="📑 Số chương trong mỗi file audio",
                                                  info="1 = mỗi chương một file. Ví dụ 5: sách được chia thành các file, mỗi file 5 chương (001-005, 006-010, …).")
                        book_pause = gr.Slider(0.0, 3.0, value=0.35, step=0.05, label="⏸️ Ngắt nghỉ giữa các đoạn (giây)",
                                               info="Chèn thẻ như [0.8] hoặc [2] vào văn bản để dừng đúng số giây đó (tối đa 10; [10][10] là 20 giây).")
                        with gr.Accordion("⏸️ Ngắt nghỉ theo từng dấu câu (như app Android)", open=False):
                            gr.Markdown("Để **0** là giữ nhịp tự nhiên của giọng đọc. Đặt số giây để dừng **đúng** chừng đó sau dấu câu "
                                        "(trong ngoặc là mức tự nhiên để tham khảo). Dấu nào được đặt thì văn bản được tách tại đó nên tạo audio chậm hơn một chút.")
                            from apps.book_batch import PAUSE_MARKS as _PAUSE_MARKS
                            _mark_names = {".": "Dấu chấm", ",": "Dấu phẩy", "?": "Dấu hỏi", "!": "Dấu chấm than", ":": "Dấu hai chấm", ";": "Dấu chấm phẩy", "…": "Dấu ba chấm"}
                            book_mark_keys = list(_PAUSE_MARKS)
                            with gr.Row():
                                book_marks = [gr.Slider(0.0, 5.0, value=0.0, step=0.1, label=f"{_mark_names[m]} {m} (tự nhiên ≈ {_PAUSE_MARKS[m]} s)", min_width=220)
                                              for m in book_mark_keys]
                        with gr.Row():
                            btn_book_list = gr.Button("🔎 Xem danh sách phần", variant="secondary")
                            btn_book_run = gr.Button("📚 Tạo audio cho cả sách", variant="primary", interactive=False)
                            btn_book_stop = gr.Button("⏹️ Dừng", variant="stop")
                        book_log = gr.Textbox(label="Tiến trình", lines=12, max_lines=24, interactive=False, autoscroll=True)

                    # --- TAB 4: VOICE CLONING (v3 Turbo / Nano, v2 GPU) ---
                    # Shown whenever the selected backbone can clone (load_model / on_backbone_change
                    # keep it in sync); the default backbone is v3 Turbo, so it is visible from the start.
                    with gr.Tab("🎤 Voice Cloning", id="clone_tab", visible=_supports_cloning(default_backbone)) as tab_custom:
                        # Initial clone-tab state must match the DEFAULT backbone:
                        # on_backbone_change only fires when the dropdown changes, so a
                        # v2-GPU default would otherwise keep v3's "no transcript" copy
                        # and a hidden reference-text box (-> false "missing ref text").
                        _default_is_v2_gpu = (default_backbone == "VieNeu-TTS-v2 (GPU)")
                        clone_info_md = gr.Markdown(
                            "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                            "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                            "reference transcript để clone giọng."
                            if _default_is_v2_gpu else
                            "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                            "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
                        )
                        with gr.Group(visible=True) as cloning_elements_group:
                            custom_audio = gr.Audio(label="Audio giọng mẫu (3-5 giây) (.wav)", type="filepath")
                            cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                            denoise_checkbox = gr.Checkbox(
                                value=True, label="🔇 Denoise audio mẫu",
                                info="Khử nhiễu nền + chuẩn hoá audio mẫu trước khi clone (khuyến nghị). Audio dài hơn 8 giây sẽ được cắt ngắn.",
                            )
                            # v3 clones from audio only — the reference transcript box AND
                            # its transcript-bearing example table live in one group that is
                            # hidden for v3 (toggled by on_backbone_change). gr.Examples keeps
                            # its own Dataset copy of the columns, so hiding only the textbox
                            # would leave the transcript column on screen (issue #191).
                            _ref_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref")
                            _ref_examples = [
                                [os.path.join(_ref_dir, "example.wav"), "Ví dụ 2. Tính trung bình của dãy số."],
                                [os.path.join(_ref_dir, "example_2.wav"), "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện."],
                                [os.path.join(_ref_dir, "example_3.wav"), "Cậu có nhìn thấy không?"],
                                [os.path.join(_ref_dir, "example_4.wav"), "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."],
                            ]
                            with gr.Group(visible=_default_is_v2_gpu) as v2_ref_text_group:
                                custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)")
                                gr.Examples(
                                    examples=_ref_examples,
                                    inputs=[custom_audio, custom_text],
                                    label="Ví dụ mẫu để thử nghiệm clone giọng"
                                )
                            # v3: audio-only examples (no transcript column).
                            with gr.Group(visible=not _default_is_v2_gpu) as v3_ref_examples_group:
                                gr.Examples(
                                    examples=[[row[0]] for row in _ref_examples],
                                    inputs=[custom_audio],
                                    label="Ví dụ mẫu để thử nghiệm clone giọng (v3 chỉ cần audio)"
                                )

                            gr.Markdown("""
                            **💡 Mẹo nhỏ:** Nếu kết quả Zero-shot Voice Cloning chưa như ý, bạn hãy cân nhắc **Finetune (LoRA)** để đạt chất lượng tốt nhất. 
                            Hướng dẫn chi tiết có tại file: `finetune/README.md` hoặc xem trên [GitHub](https://github.com/pnnbao97/VieNeu-TTS/tree/main/finetune).
                            """)

                        clone_text_input = gr.Textbox(
                            label="Văn bản đọc thử bằng giọng vừa clone",
                            lines=4,
                            value=DEFAULT_CLONE_TEXT,
                        )
                        btn_generate_clone = gr.Button("🎵 Tạo giọng nói", variant="primary", interactive=False)

                        with gr.Accordion("💾 Lưu giọng này vào danh sách giọng mẫu", open=True):
                            gr.Markdown(
                                "Đặt tên rồi bấm **Lưu giọng**: giọng sẽ có trong danh sách giọng mẫu của "
                                "**Đọc truyện, Hội thoại, SRT** và được giữ lại cho những lần mở app sau "
                                "(lưu trong thư mục `~/.vieneu`). Chỉ VieNeu v3 (Turbo / Nano) lưu được."
                            )
                            # Inputs on one line, the action button on its own line at its
                            # natural size: a button stretched to the height of two labelled
                            # textboxes looks like a slab and throws the row off balance.
                            with gr.Row():
                                clone_save_name = gr.Textbox(label="Tên giọng", placeholder="VD: Anh Tuấn", scale=2)
                                clone_save_desc = gr.Textbox(label="Mô tả (tuỳ chọn)", placeholder="nam, trầm, kể chuyện", scale=3)
                            btn_save_voice = gr.Button("💾 Lưu giọng", variant="secondary", size="sm", scale=0, min_width=160)
                            clone_save_status = gr.Markdown(visible=False)
                            # Saved voices: a label-less, container-less dropdown is exactly one
                            # control tall, so a small delete button lines up with it.
                            gr.Markdown("**Giọng đã lưu**", elem_classes="field-caption")
                            with gr.Row(elem_classes="inline-row"):
                                user_voice_dd = gr.Dropdown(choices=[], value=None, show_label=False, container=False, scale=4)
                                btn_delete_voice = gr.Button("🗑️ Xoá", variant="secondary", size="sm", scale=0, min_width=110)

                        with gr.Accordion("🔬 Tìm đoạn mẫu tốt nhất từ một bản thu dài (v3 Turbo)", open=False):
                            gr.Markdown(
                                "Bản clone chỉ dùng **8 giây** của audio mẫu, và 8 giây đó quyết định cả chất giọng lẫn cách đọc. "
                                "Tải lên một bản thu **30 giây đến vài phút** (một người, đọc đúng kiểu bạn muốn nghe): app tự cắt ra vài đoạn 8 giây "
                                "nhiều tiếng nói nhất, clone từng đoạn, cho mỗi bản clone đọc cùng một câu và chấm **độ giống** với giọng trong cả bản thu. "
                                "Điểm số chỉ để xếp thứ tự; hãy nghe rồi chọn."
                            )
                            lab_audio = gr.Audio(label="Bản thu dài của một giọng (.wav / .mp3)", type="filepath")
                            with gr.Row():
                                lab_count = gr.Slider(2, LAB_MAX, value=4, step=1, label="Số đoạn thử")
                                lab_both = gr.Checkbox(value=False, label="Thử cả bật và tắt khử nhiễu", info="Gấp đôi số bản clone; hữu ích khi bản thu vốn đã sạch.")
                            btn_lab = gr.Button("🔬 Clone và so sánh", variant="primary")
                            lab_status = gr.Markdown()
                            lab_players = [gr.Audio(visible=False, interactive=False) for _ in range(LAB_MAX)]
                            lab_state = gr.State([])
                            with gr.Row():
                                lab_pick = gr.Dropdown(choices=[], value=None, label="Bản muốn lưu", scale=3)
                                lab_name = gr.Textbox(label="Tên giọng", placeholder="VD: Giọng tôi (bản 2)", scale=2)
                            btn_lab_save = gr.Button("💾 Lưu bản này thành giọng mẫu", variant="secondary", size="sm", scale=0, min_width=220)
                            lab_save_status = gr.Markdown(visible=False)

                        with gr.Accordion("📱 Dùng giọng trên app điện thoại (Doc truyen)", open=False) as phone_acc:
                            gr.Markdown(
                                "App điện thoại đọc bằng model **v3 Nano**, nên giọng ở đây (giọng clone, giọng có sẵn hay giọng đã fine-tune) "
                                "được chuyển sang: giọng đọc một đoạn mẫu, rồi Nano học lại đoạn đó. Bấm tạo, **nghe thử bản trên điện thoại**, "
                                "rồi chép file `.dtvoice` sang điện thoại (cáp USB, Google Drive, Zalo…) và mở nó trong app: "
                                "**Giọng đọc → Từ tệp**. Lần đầu máy tải model Nano của điện thoại (khoảng 380 MB).\n\n"
                                "Nano nhỏ hơn Turbo nhiều, nên giọng trên điện thoại chỉ gần giống chứ không y hệt. "
                                "Muốn nghe đúng giọng Turbo trên điện thoại, hãy tạo audio cho cả sách ở tab **📚 Sách → audio** rồi thêm vào app."
                            )
                            with gr.Row():
                                phone_voice_dd = gr.Dropdown(choices=[], value=None, label="Giọng", allow_custom_value=True, scale=3)
                                phone_name = gr.Textbox(label="Tên trên điện thoại", placeholder="VD: Giọng bố", scale=2)
                            btn_phone = gr.Button("📱 Tạo file giọng cho điện thoại", variant="primary")
                            phone_status = gr.Markdown()
                            with gr.Row():
                                phone_source = gr.Audio(label="Giọng gốc (Turbo) đọc đoạn mẫu", interactive=False)
                                phone_sample = gr.Audio(label="Trên điện thoại (Nano) sẽ nghe như thế này", interactive=False)
                            phone_file = gr.File(label="File giọng (.dtvoice)", interactive=False)

                    # --- TAB: FINE-TUNE (record a dataset sentence by sentence; training runs outside the app) ---
                    with gr.Tab("🎓 Fine-tune giọng", id="finetune_tab"):
                        gr.Markdown(
                            "Fine-tune dạy model **một giọng** từ 10–30 phút thu âm, bám giọng sát hơn hẳn clone 8 giây. Làm ba bước:\n\n"
                            "1. **Thu âm ở đây**: app hiện từng câu, bạn đọc và lưu. Nên có **100–300 câu** (mỗi câu 3–15 giây), phòng yên tĩnh, "
                            "cùng một micro, đọc đúng kiểu bạn muốn model đọc sách. Có thể nghỉ và thu tiếp vào hôm khác.\n"
                            "2. **Đóng app này**, rồi chạy `finetune_windows.bat \"Tên giọng\"` (card 6 GB không chứa nổi cả app lẫn việc train).\n"
                            "3. **Mở lại app**: model mới xuất hiện trong danh sách model với tên có 🎓."
                        )
                        with gr.Row():
                            ft_name = gr.Textbox(label="Tên giọng", placeholder="VD: Giong toi", scale=2)
                            ft_source = gr.Textbox(label="Lấy câu từ sách hoặc văn bản (.epub / .txt); để trống = 40 câu có sẵn", scale=3,
                                                   placeholder=r"VD: D:\Sach\Truyen.epub (đọc chính loại văn bạn muốn nghe)")
                        btn_ft_start = gr.Button("▶ Bắt đầu / thu tiếp", variant="secondary")
                        ft_progress = gr.Markdown()
                        ft_sentence = gr.Textbox(label="Hãy đọc câu này", lines=3, interactive=False)
                        ft_audio = gr.Audio(label="Bản thu của câu này (thu bằng micro, hoặc tải lên file ghi âm của câu này)", sources=["microphone", "upload"], type="filepath")
                        with gr.Row():
                            btn_ft_save = gr.Button("💾 Lưu và sang câu tiếp", variant="primary")
                            btn_ft_skip = gr.Button("⏭ Bỏ qua câu này", variant="secondary")
                        with gr.Accordion("📲 Tải lên một bản ghi âm dài (đọc nhiều câu liền, ví dụ bằng điện thoại)", open=False):
                            gr.Markdown(
                                "Ghi âm bằng điện thoại cũng được (file .m4a, .mp3, .wav…). Bấm **Bắt đầu / thu tiếp**, rồi đọc **đúng thứ tự** các câu "
                                "bắt đầu từ câu đang hiện ở trên. Đọc liền mạch trong câu, **ngừng khoảng một giây sau mỗi câu**. Muốn bỏ một câu, đừng đọc nó "
                                "mà hãy dừng ghi âm ở câu trước. Khi tải lên, app cắt bản ghi tại các chỗ ngừng và gán lần lượt cho từng câu; nếu một đoạn "
                                "không khớp với câu của nó (ví dụ quên ngừng, đọc sót), app không lưu gì và báo đoạn sai."
                            )
                            ft_long = gr.Audio(label="Bản ghi âm dài", sources=["upload"], type="filepath")
                            btn_ft_long = gr.Button("📲 Cắt và lưu theo từng câu", variant="secondary")
                        with gr.Accordion("📱 Nhập bản thu từ điện thoại (file .dtdata của app Doc truyen)", open=False):
                            gr.Markdown(
                                "Trong app điện thoại: **Giọng đọc → Bản thu để fine-tune trên máy tính → Thu âm từng câu**, rồi **Xuất** ra file `.dtdata` "
                                "(lưu vào Tải xuống hoặc Google Drive) và chép sang máy tính. Bản thu giọng khi clone trên điện thoại cũng nằm trong file đó. "
                                "Ô **Tên giọng** ở trên để trống thì dùng tên trong file."
                            )
                            ft_dtdata = gr.File(label="File .dtdata", file_types=[".dtdata", ".zip"], type="filepath")
                            btn_ft_dtdata = gr.Button("📱 Thêm các bản thu này", variant="secondary")
                        ft_state = gr.State({"sentences": [], "at": 0})
                        ft_train_md = gr.Markdown()

                # Global Generation Settings
                with gr.Row():
                    use_batch = gr.Checkbox(
                        value=True, 
                        label="⚡ Batch Processing",
                        info="Gộp nhiều đoạn vào một lần forward. v3 Turbo: cần GPU CUDA (không cần LMDeploy). v1/v2: cần GPU + LMDeploy. Trên CPU/ONNX tuỳ chọn này không có tác dụng."
                    )
                    max_batch_size_run = gr.Slider(
                        minimum=1,
                        maximum=64,
                        value=default_batch_size,
                        step=1,
                        label="📊 Batch Size (Generation)",
                        info="Số lượng đoạn văn bản xử lý cùng lúc. Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi Out of Memory."
                    )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao (Generation)", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=default_temp, step=0.1,
                            label="🌡️ Temperature", 
                            info="Độ sáng tạo. Cao = đa dạng cảm xúc hơn nhưng dễ lỗi. Thấp = ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512,
                            value=256, step=32,
                            label="📝 Max Chars per Chunk",
                            info="Độ dài tối đa mỗi đoạn xử lý (mặc định: 256)."
                        )
                
                # synthesize_speech(mode_tab=...): the story tab always reads a preset,
                # the Voice Cloning tab always clones from the uploaded sample.
                current_mode_state = gr.State("preset_mode")
                clone_mode_state = gr.State("custom_mode")
                
                with gr.Row():
                    btn_stop = gr.Button("⏹️ Dừng", variant="stop", scale=1, interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True
                )
                with gr.Group():
                    status_output = gr.Textbox(
                        label="Trạng thái", 
                        elem_classes="status-box",
                        lines=2,
                        max_lines=10,
                        show_copy_button=True
                    )
                with gr.Group():
                    estimate_output = gr.Textbox(
                        label="Ước tính thời gian",
                        elem_classes="estimate-box",
                        lines=2,
                        max_lines=4,
                        show_copy_button=True
                    )
                download_btn = gr.DownloadButton(
                    "📥 Tải xuống file Audio",
                    variant="primary",
                    visible=False,
                    elem_classes="download-btn"
                )
                # Only claim a watermark when the real watermarker (resemble-perth) is importable —
                # otherwise the SDK silently skips watermarking (issue #191).
                gr.Markdown(
                    "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 Audio được đóng dấu bản quyền ẩn (Perth Watermarker) để định danh AI.</div>",
                    visible=_watermark_available(),
                )
        
        # ONNX codecs (v2 CPU) cannot clone: hide the tab; v3 / v2 GPU keep it.
        codec_select.change(
            lambda codec, bb: gr.update(visible=_supports_cloning(bb) and "onnx" not in (codec or "").lower()),
            inputs=[codec_select, backbone_select],
            outputs=[tab_custom],
        )

        custom_audio.change(validate_audio_duration, inputs=[custom_audio], outputs=[cloning_warning_msg])
        
        # --- Custom Model Event Handlers ---

        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            is_v3 = "v3" in (choice or "").lower()
            is_v2_gpu = (choice == "VieNeu-TTS-v2 (GPU)")
            # Voice Cloning: v3 clones from audio only; v2 (GPU) clones from
            # audio + a reference transcript. Both expose the cloning tab.
            clone_ok = is_v3 or is_v2_gpu
            print(f"   🔄 Backbone changed to: {choice}")
            
            # 1. Device logic
            # Allow hardware acceleration (MPS/CUDA/Auto) for all GPU models AND Turbo (GGUF) models
            is_hw_accel_supported = "(GPU)" in choice or "v2-Turbo" in choice or "v3" in choice.lower() or is_custom
            
            if is_hw_accel_supported:
                dev_choices = get_available_devices()
                initial_dev = "Auto"
            else:
                dev_choices = ["CPU"]
                initial_dev = "CPU"
            
            # 2. Parameter logic
            if is_v3:
                # v3 Turbo uses its own MOSS codec (PyTorch); 0.8 khớp bản tham chiếu.
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update()  # ô văn bản để trống, không điền mẫu
                temp_update = gr.update(value=0.8)
            elif "Turbo" in choice:
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update()  # ô văn bản để trống, không điền mẫu
                temp_update = gr.update(value=0.4)
            elif "(CPU)" in choice:
                codec_update = gr.update(value="NeuCodec (ONNX)", interactive=False)
                text_update = gr.update()  # ô văn bản để trống, không điền mẫu
                temp_update = gr.update(value=0.7)
            else:
                codec_update = gr.update(value="NeuCodec (Distill)", interactive=False)
                text_update = gr.update()  # ô văn bản để trống, không điền mẫu
                temp_update = gr.update(value=0.7)

            # Reference-transcript box + info text differ between v2 and v3 clone.
            if is_v2_gpu:
                clone_info_update = gr.update(value=(
                    "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                    "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                    "reference transcript để clone giọng."
                ))
            else:
                clone_info_update = gr.update(value=(
                    "ℹ️ **Voice Cloning (VieNeu-TTS v3).** Chỉ cần tải lên audio mẫu "
                    "3–5 giây; v3 clone trực tiếp từ audio, không cần nhập nội dung."
                ))

            return (
                gr.update(visible=is_custom),
                codec_update,
                text_update,
                temp_update,
                gr.update(choices=dev_choices, value=initial_dev),
                gr.update(visible=clone_ok),   # cloning_elements_group
                gr.update(visible=clone_ok),   # tab_custom — clone tab (v3 + v2 GPU)
                gr.update(value=16 if is_v3 else 4),  # max_batch_size_run — v3 batches chunks (slider up to 64)
                gr.update(visible=not is_v3),  # use_lmdeploy_cb — irrelevant for v3 (PyTorch, no LMDeploy)
                gr.update(visible=is_v2_gpu),  # v2_ref_text_group — transcript box + transcript examples (v2 only)
                gr.update(visible=not is_v2_gpu),  # v3_ref_examples_group — audio-only examples
                clone_info_update,             # clone_info_md
                gr.update(value=256),  # max_chars_chunk_slider
            )

        backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[
                custom_model_group,
                codec_select,
                text_input,
                temperature_slider,
                device_choice,
                cloning_elements_group,
                tab_custom,
                max_batch_size_run,
                use_lmdeploy_cb,
                v2_ref_text_group,
                v3_ref_examples_group,
                clone_info_md,
                max_chars_chunk_slider,
            ]
        )
        
        custom_backbone_model_id.change(
            on_custom_id_change,
            inputs=[custom_backbone_model_id],
            outputs=[custom_backbone_base_model, custom_audio, custom_text]
        )

        load_event = btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_custom,
                     conv_tab,
                     *speaker_voice_dds]
        )

        # --- Voice Cloning tab: generate button + saved-voice list follow the model state ---
        def _user_voice_choices():
            names = list_user_voices(tts) if model_loaded else []
            return gr.update(choices=names, value=names[0] if names else None)

        def _after_model_load():
            return gr.update(interactive=bool(model_loaded)), _user_voice_choices()

        btn_load.click(lambda: gr.update(interactive=False), outputs=btn_generate_clone)
        load_event.then(_after_model_load, outputs=[btn_generate_clone, user_voice_dd])
        
        # --- PDF Upload Event Handlers ---
        def on_pdf_upload(pdf_file):
            """Extract text from uploaded PDF and populate text input."""
            if not pdf_file:
                return gr.update(), gr.update(visible=False)
            extracted = extract_text_from_pdf(pdf_file)
            if extracted.startswith("⚠️"):
                return gr.update(), gr.update(value=extracted, visible=True)
            char_count = len(extracted)
            return (
                gr.update(value=extracted),
                gr.update(
                    value=f"✅ Đã trích xuất **{char_count:,}** ký tự từ file PDF. Bạn có thể chỉnh sửa văn bản bên dưới.",
                    visible=True
                )
            )

        pdf_upload.change(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[text_input, pdf_status]
        )
        btn_extract_pdf.click(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[text_input, pdf_status]
        )

        # --- Conversation Event Handlers ---
        # Scan speakers → update all 8 slot rows/names/dropdowns
        btn_detect_speakers.click(
            fn=extract_speakers_from_script,
            inputs=[conv_script_input],
            outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
        )
        
        conv_gen_event = btn_generate_conv.click(
            fn=synthesize_conversation_with_empty_estimate,
            inputs=[conv_script_input,
                    *speaker_name_boxes,
                    *speaker_voice_dds,
                    silence_slider, temperature_slider, max_chars_chunk_slider,
                    session_id_state],
            outputs=[audio_output, status_output, estimate_output]
        )
        btn_generate_conv.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate_conv.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        conv_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Auto-adjust Temperature on Tab Switch ---
        conv_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else 1.0),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        single_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else default_temp),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        tab_custom.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else default_temp),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        
        # --- Standard Generation Handlers ---
        gen_event = btn_generate.click(
            fn=synthesize_speech_with_estimate,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state,
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider,
                    denoise_checkbox, session_id_state],
            outputs=[audio_output, status_output, estimate_output]
        )
        btn_generate.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Voice Cloning: same synthesis path, clone mode, its own text box ---
        clone_gen_event = btn_generate_clone.click(
            fn=synthesize_speech_with_estimate,
            inputs=[clone_text_input, voice_select, custom_audio, custom_text, clone_mode_state,
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider,
                    denoise_checkbox, session_id_state],
            outputs=[audio_output, status_output, estimate_output]
        )
        btn_generate_clone.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate_clone.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        clone_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Save / delete a cloned voice as a preset ---
        def _refresh_voice_caches():
            """Recompute the dropdown lists after the preset table changed."""
            global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
            try:
                voices = tts.list_preset_voices()
            except Exception:
                voices = []
            _sort_voices(tts, voices)
            PRESET_VOICES_CACHE = voices

            def _podcast(v_id):
                val = tts._preset_voices.get(v_id, {}).get("podcast", True)
                return val.strip().lower() == "true" if isinstance(val, str) else bool(val)

            CONV_VOICES_CACHE = [v for v in voices if _podcast(v[1] if isinstance(v, tuple) else v)]

        def _voice_list_updates(select_voice=None):
            """Updates for every preset dropdown: story (optionally selecting the new
            voice), SRT, and the conversation speaker slots (keep their values)."""
            choices = with_switches(PRESET_VOICES_CACHE)
            story = gr.update(choices=choices, value=select_voice) if select_voice else gr.update(choices=choices)
            return [story, gr.update(choices=choices), *([gr.update(choices=CONV_VOICES_CACHE)] * MAX_SPEAKERS)]

        def _save_voice(audio_path, name, desc, denoise):
            no_change = [gr.update()] * (2 + MAX_SPEAKERS)
            if not model_loaded or tts is None:
                return [gr.update(value="⚠️ Vui lòng tải model trước!", visible=True), gr.update(), *no_change]
            if not supports_saving(tts):
                return [gr.update(value="⚠️ Chỉ VieNeu v3 (Turbo / Nano) mới lưu được giọng.", visible=True), gr.update(), *no_change]
            try:
                v_id = save_user_voice(tts, name, audio_path, denoise=bool(denoise), description=desc)
            except Exception as e:  # noqa: BLE001
                return [gr.update(value=f"❌ {e}", visible=True), gr.update(), *no_change]
            _refresh_voice_caches()
            msg = f"✅ Đã lưu giọng **{v_id}** — đã có trong danh sách giọng mẫu của Đọc truyện, Hội thoại và SRT."
            return [gr.update(value=msg, visible=True), _user_voice_choices(), *_voice_list_updates(select_voice=v_id)]

        btn_save_voice.click(
            fn=_save_voice,
            inputs=[custom_audio, clone_save_name, clone_save_desc, denoise_checkbox],
            outputs=[clone_save_status, user_voice_dd, voice_select, srt_voice, *speaker_voice_dds],
        )

        def _lab_run(audio_path, count, both):
            hidden = [gr.update(visible=False)] * LAB_MAX
            if not model_loaded or tts is None:
                return ["⚠️ Vui lòng nạp model trước!", [], gr.update(choices=[], value=None), *hidden]
            if not audio_path:
                return ["⚠️ Hãy tải lên một bản thu.", [], gr.update(choices=[], value=None), *hidden]
            from apps.voice_lab import compare, label
            try:
                results = compare(tts, audio_path, tempfile.mkdtemp(prefix="vieneu-lab-"), count=int(count), both_denoise=bool(both))[:LAB_MAX]
            except Exception as e:  # noqa: BLE001
                return [f"❌ {e}", [], gr.update(choices=[], value=None), *hidden]
            labels = [label(i, r) for i, r in enumerate(results, 1)]
            players = [gr.update(value=r["sample"], label=text, visible=True) for r, text in zip(results, labels)]
            status = (f"Đã clone {len(results)} bản, xếp từ giống nhất. Mỗi bản đọc cùng một đoạn văn: hãy nghe và chọn bản bạn thích, "
                      "rồi đặt tên và lưu ở dưới.")
            return [status, results, gr.update(choices=labels, value=labels[0]), *players, *hidden[len(players):]]

        btn_lab.click(lambda: "⏳ Đang clone và đọc thử từng đoạn…", outputs=[lab_status]).then(
            _lab_run, inputs=[lab_audio, lab_count, lab_both], outputs=[lab_status, lab_state, lab_pick, *lab_players])

        def _lab_save(results, picked, name):
            no_change = [gr.update()] * (2 + MAX_SPEAKERS)
            from apps.voice_lab import label
            chosen = next((r for i, r in enumerate(results or [], 1) if label(i, r) == picked), None)
            if chosen is None:
                return [gr.update(value="⚠️ Hãy chạy so sánh và chọn một bản trước.", visible=True), gr.update(), *no_change]
            try:
                v_id = save_user_voice(tts, name, chosen["clip"], denoise=bool(chosen["denoise"]))
            except Exception as e:  # noqa: BLE001
                return [gr.update(value=f"❌ {e}", visible=True), gr.update(), *no_change]
            _refresh_voice_caches()
            return [gr.update(value=f"✅ Đã lưu giọng **{v_id}**.", visible=True), _user_voice_choices(), *_voice_list_updates(select_voice=v_id)]

        btn_lab_save.click(_lab_save, inputs=[lab_state, lab_pick, lab_name],
                           outputs=[lab_save_status, user_voice_dd, voice_select, srt_voice, *speaker_voice_dds])

        # --- A voice for the phone app ---
        def _phone_voices():
            if not (model_loaded and tts is not None and hasattr(tts, "get_preset_voice")):
                return gr.update(choices=[], value=None)
            current = PRESET_VOICES_CACHE[0] if PRESET_VOICES_CACHE else None
            value = current[1] if isinstance(current, (list, tuple)) else current
            return gr.update(choices=with_switches(PRESET_VOICES_CACHE), value=value)

        phone_acc.expand(_phone_voices, outputs=[phone_voice_dd])
        # Another model has other voices (a fine-tuned model has only its own): follow every load, not only the opening.
        load_event.then(_phone_voices, outputs=[phone_voice_dd])

        def _phone_export(voice_choice, name):
            if not model_loaded or tts is None:
                yield "⚠️ Vui lòng nạp model trước!", None, None, None
                return
            if "v3" not in (current_backbone or "").lower() or "nano" in (current_backbone or "").lower():
                yield "⚠️ Hãy nạp một model **v3 Turbo** (bản thường hoặc bản 🎓 đã fine-tune).", None, None, None
                return
            from apps.phone_voice import export
            voice_id = resolve_voice_id(voice_choice) if voice_choice else None
            log = []
            done = threading.Event()
            result, error = {}, []

            def work():
                try:
                    result.update(export(tts, voice_id, (name or "").strip() or voice_id or "Giọng của tôi",
                                         Path.home() / "Music" / "Doc truyen - giong", report=log.append))
                except BaseException as e:  # noqa: BLE001
                    error.append(e)
                finally:
                    done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.wait(1.0):
                yield "⏳ " + (log[-1] if log else "Đang chuẩn bị…"), None, None, None
            if error:
                yield f"❌ {error[0]}", None, None, None
                return
            yield (f"✅ Đã tạo `{result['file']}`. Nghe hai bản bên dưới; nếu ưng, chép file này sang điện thoại và mở trong app "
                   "(Giọng đọc → Từ tệp)."), result["source"], result["sample"], result["file"]

        btn_phone.click(_phone_export, inputs=[phone_voice_dd, phone_name], outputs=[phone_status, phone_source, phone_sample, phone_file])

        # --- Fine-tune: record one sentence at a time ---
        def _ft_view(name, state, note=""):
            from apps import finetune_studio as ft
            try:
                count, minutes, done = ft.progress(name)
                safe = ft.safe_name(name)
            except ValueError as e:
                return [state, "", f"⚠️ {e}", gr.update(), ""]
            sentences, at = state.get("sentences", []), int(state.get("at", 0))
            while at < len(sentences) and " ".join(sentences[at].split()) in done:
                at += 1
            state = {"sentences": sentences, "at": at}
            left = len(sentences) - at
            lines = [note] if note else []
            lines.append(f"**Đã thu {count} câu · {minutes:.1f} phút** (nên đạt 10–30 phút). Còn {max(left, 0)} câu trong danh sách này.")
            if sentences and left <= 0:
                lines.append("Đã hết câu trong danh sách. Muốn thu thêm, nhập đường dẫn một cuốn sách ở trên rồi bấm **Bắt đầu / thu tiếp**.")
            train = (f"**Khi đã đủ dữ liệu:** đóng cửa sổ app này, mở thư mục `{Path(__file__).resolve().parents[1]}` và chạy "
                     f"`finetune_windows.bat \"{safe}\"`. Dữ liệu nằm ở `{ft.dataset_dir(name)}`.") if count else ""
            return [state, sentences[at] if 0 <= at < len(sentences) else "", "\n\n".join(lines), None, train]

        def _ft_start(name, source, state):
            from apps import finetune_studio as ft
            try:
                ft.safe_name(name)
                sentences = ft.load_script(source)
            except Exception as e:  # noqa: BLE001
                return [state, "", f"⚠️ {e}", gr.update(), ""]
            return _ft_view(name, {"sentences": sentences, "at": 0})

        def _ft_save(name, state, audio_path):
            from apps import finetune_studio as ft
            sentences, at = state.get("sentences", []), int(state.get("at", 0))
            if not (0 <= at < len(sentences)):
                return _ft_view(name, state, "⚠️ Bấm **Bắt đầu / thu tiếp** để lấy câu.")
            try:
                seconds = ft.save_take(name, sentences[at], audio_path)
            except Exception as e:  # noqa: BLE001
                view = _ft_view(name, state, f"⚠️ {e}")
                view[3] = gr.update()  # keep the take on screen so it can be heard or redone
                return view
            return _ft_view(name, state, f"✅ Đã lưu ({seconds:.1f} giây).")

        def _ft_long(name, state, audio_path):
            from apps import finetune_studio as ft
            sentences, at = state.get("sentences", []), int(state.get("at", 0))
            if not (0 <= at < len(sentences)):
                return _ft_view(name, state, "⚠️ Bấm **Bắt đầu / thu tiếp** để lấy danh sách câu trước.")
            if not audio_path:
                return _ft_view(name, state, "⚠️ Hãy tải lên một bản ghi âm.")
            try:
                ft.safe_name(name)
                count, minutes, skipped = ft.import_reading(name, sentences[at:], audio_path)
            except Exception as e:  # noqa: BLE001
                return _ft_view(name, state, f"⚠️ {e}")
            note = f"✅ Đã cắt và lưu {count} câu ({minutes:.1f} phút)."
            if skipped:
                note += f" Bỏ qua đoạn {', '.join(map(str, skipped))}: ngắn hơn 3 hoặc dài hơn 15 giây (đọc lại các câu đó riêng)."
            return _ft_view(name, state, note)

        btn_ft_long.click(_ft_long, inputs=[ft_name, ft_state, ft_long], outputs=[ft_state, ft_sentence, ft_progress, ft_audio, ft_train_md])

        def _ft_dtdata(name, state, path):
            from apps import finetune_studio as ft
            if not path:
                return [state, gr.update(), "⚠️ Hãy chọn một file .dtdata.", gr.update(), "", gr.update()]
            try:
                result = ft.import_dataset(path, (name or "").strip() or None)
            except Exception as e:  # noqa: BLE001
                return [state, gr.update(), f"⚠️ {e}", gr.update(), "", gr.update()]
            note = f"✅ Đã thêm {result['saved']} bản thu ({result['minutes']:.1f} phút) cho giọng **{result['name']}**."
            if result["problems"]:
                note += "\n\nKhông dùng được:\n" + "\n".join(f"- {p}" for p in result["problems"][:20])
            return [*_ft_view(result["name"], state, note), result["name"]]

        btn_ft_dtdata.click(_ft_dtdata, inputs=[ft_name, ft_state, ft_dtdata],
                            outputs=[ft_state, ft_sentence, ft_progress, ft_audio, ft_train_md, ft_name])

        def _ft_skip(name, state):
            return _ft_view(name, {"sentences": state.get("sentences", []), "at": int(state.get("at", 0)) + 1})

        _ft_outputs = [ft_state, ft_sentence, ft_progress, ft_audio, ft_train_md]
        btn_ft_start.click(_ft_start, inputs=[ft_name, ft_source, ft_state], outputs=_ft_outputs)
        btn_ft_save.click(_ft_save, inputs=[ft_name, ft_state, ft_audio], outputs=_ft_outputs)
        btn_ft_skip.click(_ft_skip, inputs=[ft_name, ft_state], outputs=_ft_outputs)

        def _delete_voice(name):
            no_change = [gr.update()] * (2 + MAX_SPEAKERS)
            if not name:
                return [gr.update(value="⚠️ Chọn giọng cần xoá.", visible=True), gr.update(), *no_change]
            try:
                delete_user_voice(tts, name)
            except Exception as e:  # noqa: BLE001
                return [gr.update(value=f"❌ {e}", visible=True), gr.update(), *no_change]
            _refresh_voice_caches()
            return [gr.update(value=f"🗑️ Đã xoá giọng **{name}**.", visible=True), _user_voice_choices(), *_voice_list_updates()]

        btn_delete_voice.click(
            fn=_delete_voice,
            inputs=[user_voice_dd],
            outputs=[clone_save_status, user_voice_dd, voice_select, srt_voice, *speaker_voice_dds],
        )

        # --- Stop Button ---
        def request_stop():
            print("🛑 STOP REQUESTED via button click.")
            _STOP_EVENT.set()
            return None, "⏹️ Đã dừng tạo giọng nói.", "", gr.update(interactive=False)

        # Handler: set stop event + update UI
        # Note: We avoid cancels= here to prevent internal Gradio KeyError crashes,
        # relying instead on the frequent _STOP_EVENT.is_set() checks in the code.
        btn_stop.click(fn=request_stop, outputs=[audio_output, status_output, estimate_output, btn_stop])

        # --- Download Button Event Handlers ---
        def on_audio_generated(audio_path):
            """Show download button when audio is generated."""
            if audio_path and os.path.exists(audio_path):
                return gr.update(value=audio_path, visible=True)
            return gr.update(visible=False)

        # Connect to generation events (must be after gen_event/conv_gen_event are defined)
        gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        conv_gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        clone_gen_event.then(fn=on_audio_generated, inputs=[audio_output], outputs=[download_btn])

        # --- SRT → speech ---
        def _srt_voices():
            """The loaded model's presets for the SRT tab, fetched when the tab
            opens so the tab needs no plumbing into the model-load handler."""
            if not model_loaded or tts is None:
                return gr.update(choices=[], value=None), gr.update(interactive=False)
            try:
                voices = tts.list_preset_voices()
            except Exception:
                voices = []
            default_v = getattr(tts, "_default_voice", None)
            values = [v[1] if isinstance(v, tuple) else v for v in voices]
            if default_v not in values and values:
                default_v = values[0]
            return gr.update(choices=with_switches(voices), value=default_v), gr.update(interactive=bool(values) and hasattr(tts, "infer_batch"))

        srt_tab.select(_srt_voices, outputs=[srt_voice, btn_generate_srt])

        # --- Book → audio ---
        def _book_voices():
            voices_update, _ = _srt_voices()
            return voices_update, gr.update(interactive=bool(model_loaded and tts is not None and hasattr(tts, "infer")))

        book_tab.select(_book_voices, outputs=[book_voice, btn_book_run])

        # --- A voice of another model (a fine-tuned voice, or back to the built-in ones): load its model ---
        _LOAD_OUTPUTS = 8 + MAX_SPEAKERS

        def _switch_model(choice, codec, device, lmdeploy):
            if not (isinstance(choice, str) and choice.startswith(FT_SWITCH)):
                return
            label = choice[len(FT_SWITCH):]
            if label not in BACKBONE_CONFIGS:
                return
            idle = [gr.update()] * 7
            for out in load_model(label, codec, device, lmdeploy):
                yield (gr.update(value=label), *tuple(out)[:_LOAD_OUTPUTS], *idle)
            # The other voice lists follow the new model.
            yield (gr.update(), *([gr.update()] * _LOAD_OUTPUTS), *_after_model_load(), _phone_voices(), *_srt_voices(), *_book_voices())

        for _dd in (voice_select, book_voice, srt_voice, phone_voice_dd):
            _dd.input(_switch_model, inputs=[_dd, codec_select, device_choice, use_lmdeploy_cb],
                      outputs=[backbone_select, model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                               tab_custom, conv_tab, *speaker_voice_dds,
                               btn_generate_clone, user_voice_dd, phone_voice_dd, srt_voice, btn_generate_srt, book_voice, btn_book_run])

        def _book_source(path_text, upload):
            path = (path_text or "").strip().strip('"') or upload
            if not path:
                raise ValueError("Hãy nhập đường dẫn hoặc kéo thả một file .epub / .txt.")
            if not os.path.exists(path):
                raise ValueError(f"Không tìm thấy: {path}")
            return path

        def _book_list(path_text, upload):
            from apps.book_batch import collect_parts
            try:
                parts = collect_parts(_book_source(path_text, upload))
            except Exception as e:  # noqa: BLE001
                return f"❌ {e}"
            total = sum(len(p.text) for p in parts)
            lines = [f"{i:4d}. {p.title}  ({len(p.text):,} ký tự)" for i, p in enumerate(parts, 1)]
            return "\n".join([f"{len(parts)} phần, {total:,} ký tự:"] + lines)

        btn_book_list.click(_book_list, inputs=[book_source, book_upload], outputs=[book_log])

        def _book_run(path_text, upload, out_dir, voice_choice, fmt, paragraph_pause=0.35, chapters_per_file=1, *mark_seconds):
            from apps.book_batch import convert_book, default_output_dir
            if not model_loaded or tts is None:
                yield "⚠️ Vui lòng tải model trước!"
                return
            try:
                source = _book_source(path_text, upload)
            except Exception as e:  # noqa: BLE001
                yield f"❌ {e}"
                return
            _STOP_EVENT.clear()
            # A dropped file lives in Gradio's temp folder; write next to it only if asked.
            target = (out_dir or "").strip().strip('"') or (None if path_text else str(Path.home() / "Music" / f"{Path(source).stem} - audio"))
            rules = {m: float(s) for m, s in zip(book_mark_keys, mark_seconds) if s and float(s) > 0}
            log: list[str] = []
            done = threading.Event()
            error: list[BaseException] = []

            def work():
                try:
                    convert_book(tts, source, target, resolve_voice_id(voice_choice) if voice_choice else None,
                                 fmt, stop=_STOP_EVENT, report=log.append, paragraph_pause=float(paragraph_pause),
                                 chapters_per_file=int(chapters_per_file or 1), pause_rules=rules)
                except BaseException as e:  # noqa: BLE001
                    error.append(e)
                finally:
                    done.set()

            threading.Thread(target=work, daemon=True).start()
            log.append(f"📁 Lưu vào: {target or default_output_dir(Path(source))}")
            while not done.wait(1.0):
                yield "\n".join(log[-200:])
            if error:
                log.append(f"❌ Lỗi: {error[0]}")
            yield "\n".join(log[-200:])

        btn_book_run.click(_book_run, inputs=[book_source, book_upload, book_out, book_voice, book_format, book_pause, book_chapters, *book_marks],
                           outputs=[book_log])
        btn_book_sample.click(voice_sample, inputs=[book_voice], outputs=[book_sample_audio])
        btn_voice_sample.click(voice_sample, inputs=[voice_select], outputs=[voice_sample_audio])
        btn_book_stop.click(lambda: _STOP_EVENT.set(), outputs=None)

        def _srt_run(srt_path, voice_choice, keep_timing, fmt, use_batch_flag, batch_size):
            _STOP_EVENT.clear()
            # Same Batch Size slider as the other tabs (GPU only; CPU goes cue by cue).
            bs = max(1, int(batch_size)) if use_batch_flag else 1
            yield from srt_to_speech(
                tts, srt_path, resolve_voice_id(voice_choice), bool(keep_timing), fmt,
                stop_requested=_STOP_EVENT.is_set, batch_size=bs,
            )

        srt_gen_event = btn_generate_srt.click(
            fn=wrap_with_estimate(_srt_run),
            inputs=[srt_file, srt_voice, srt_keep_timing, srt_format, use_batch, max_batch_size_run],
            outputs=[audio_output, status_output, estimate_output],
        )
        btn_generate_srt.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate_srt.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        srt_gen_event.then(fn=on_audio_generated, inputs=[audio_output], outputs=[download_btn])
        # Also connect the stop button to hide download
        btn_stop.click(
            fn=lambda: gr.update(visible=False),
            outputs=[download_btn]
        )

        # Persistence: Restore UI state on load
        demo.load(
            fn=restore_ui_state,
            outputs=[model_status, btn_generate, btn_generate_conv, btn_stop]
        ).then(_after_model_load, outputs=[btn_generate_clone, user_voice_dd]).then(
            fn=autoload_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_custom, conv_tab, *speaker_voice_dds]
        ).then(_after_model_load, outputs=[btn_generate_clone, user_voice_dd]).then(_phone_voices, outputs=[phone_voice_dd])

def main():
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    # Optional login for a UI others can reach (Docker, LAN, Colab share link):
    # VIENEU_WEB_AUTH="user:password".
    auth = None
    creds = os.getenv("VIENEU_WEB_AUTH", "")
    if creds:
        user, sep, password = creds.partition(":")
        if not (user and sep and password):
            raise SystemExit("VIENEU_WEB_AUTH must look like 'user:password'")
        auth = (user, password)
    elif share or server_name not in ("127.0.0.1", "localhost", "::1"):
        where = "a public share link" if share else f"{server_name}:{server_port}"
        print(f"⚠️  The Web UI is reachable beyond this machine ({where}) with no login. "
              "Set VIENEU_WEB_AUTH=user:password to require one.")

    # Without an explicit GRADIO_SERVER_PORT, take the first free port from 7860 so a second copy
    # (or another app on 7860) does not stop this one from starting.
    if os.getenv("GRADIO_SERVER_PORT") is None:
        server_port = _first_free_port(server_name, server_port, server_port + 20)
    print(f"🌐 VieNeu-TTS: http://{'127.0.0.1' if server_name in ('0.0.0.0', '::') else server_name}:{server_port}")
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share, auth=auth,
                        inbrowser=env_bool("VIENEU_OPEN_BROWSER", default=False))


def _first_free_port(host: str, start: int, end: int) -> int:
    import socket
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1" if host in ("0.0.0.0", "localhost") else host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {start} and {end - 1}; set GRADIO_SERVER_PORT.")

if __name__ == "__main__":
    main()
