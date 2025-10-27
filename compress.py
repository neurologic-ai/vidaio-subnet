#!/usr/bin/env python3
"""
Fixed H.265/HEVC compression script — Linux-ready, high-quality, container-safe
Modifications:
  - Assume all input videos are 8-bit, yuv420p.
  - Use libx264 (H.264) when input extension is .avi.
  - Golden search scoring changed to:
       compression = initial_size / final_size
       score = 0.7 * compression + 0.3 * vmaf
    and the golden search maximizes that score.
Usage: same as before (see main()).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

sys.path.append(str(Path(__file__).parent))
try:
    from score import calculate_compression_score
except Exception:
    # keep a harmless fallback (not used by golden_search scoring anymore)
    def calculate_compression_score(**kwargs):
        vmaf_score = kwargs.get('vmaf_score', 0)
        compression_rate = kwargs.get('compression_rate', 1.0)
        return vmaf_score - compression_rate * 10, compression_rate, vmaf_score, 'fallback'


# ---- Defaults for the fast VMAF function ----
SAMPLING_VMAF_SUBSAMPLE = 8            # number of samples used by libvmaf (n_subsample)
SAMPLING_VMAF_DOWNSCALE_HALF = False   # whether to half-res during libvmaf preproc
VMAF_THREADS = 0                       # libvmaf n_threads (0 -> autodetect)


# ---- Helper: run a command via subprocess and return (code, stdout, stderr) ----
def run_cmd(args: List[str], timeout: int = 1200) -> Tuple[int, str, str]:
    """
    Runs ffmpeg (or any command if full path supplied in args[0]) with provided args.
    Prepends 'ffmpeg' if the first element doesn't look like an executable path.
    Returns (returncode, stdout, stderr).
    """
    if not args:
        raise ValueError("run_cmd requires a non-empty args list")
    cmd = args[:] if args[0].lower().endswith("ffmpeg") or os.path.basename(args[0]).lower() == "ffmpeg" else ["ffmpeg"] + args
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False
        )
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return -1, "", f"TimeoutExpired: {e}"


# ---- Helper: probe basic video properties (width, height, fps) ----
def _probe_video_props(path: str) -> Tuple[int, int, float]:
    """
    Returns (width, height, fps) for the first video stream in the file.
    On failure returns (0,0,0.0).
    """
    try:
        res = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not res.stdout:
            return 0, 0, 0.0
        info = json.loads(res.stdout)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                w = int(s.get('width') or 0)
                h = int(s.get('height') or 0)
                rfr = s.get('r_frame_rate') or s.get('avg_frame_rate') or '0/1'
                try:
                    fps = float(Fraction(rfr))
                except Exception:
                    fps = 0.0
                return w, h, fps
    except Exception:
        pass
    return 0, 0, 0.0


# ---- The fast VMAF function (self-contained) ----
def vmaf_mean_aligned_fast(ref: str,
                           dist: str,
                           src_for_norm: Optional[str] = None,
                           n_subsample: int = SAMPLING_VMAF_SUBSAMPLE,
                           half_res: bool = SAMPLING_VMAF_DOWNSCALE_HALF,
                           vmaf_threads: int = VMAF_THREADS) -> float:
    """
    Fast sampling VMAF:
      - Probe reference (or src_for_norm) to learn width/height/fps
      - Align PTS + normalize CFR (fps), optionally half-res
      - Force format to yuv420p for libvmaf (inputs are 8-bit yuv420p per assumption)
      - Run libvmaf with n_subsample and parse JSON log result
    Returns float VMAF mean (0-100). Raises on unrecoverable errors.
    """
    srcn = src_for_norm or ref

    w, h, fps = _probe_video_props(srcn)
    scale_filter = ""
    fps_filter = ""
    if w > 0 and h > 0:
        if half_res:
            w = max(1, w // 2)
            h = max(1, h // 2)
        scale_filter = f"scale={w}:{h}:flags=bicubic,"
    if fps > 0:
        fps_filter = f"fps=fps={fps},"

    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        opts = [f"n_threads={vmaf_threads}", "log_fmt=json", f"log_path={logp}"]
        if n_subsample and n_subsample > 1:
            opts.append(f"n_subsample={n_subsample}")

        lavfi = (
            f"[0:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[ref];"
            f"[1:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[dist];"
            f"[ref][dist]libvmaf=" + ":".join(opts)
        )

        args = [
            "-hide_banner",
            "-y",
            "-i", str(Path(ref).resolve()),
            "-i", str(Path(dist).resolve()),
            "-lavfi", lavfi,
            "-f", "null", "-"
        ]
        code, out, err = run_cmd(args, timeout=1200)
        if code != 0:
            raise RuntimeError(f"VMAF (fast) ffmpeg failed (code={code}): {err[-2000:]}")

        if not os.path.exists(logp):
            raise RuntimeError("VMAF JSON log not found after libvmaf run")
        try:
            with open(logp, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read/parse vmaf JSON log: {e}")

        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            frames = data.get("frames", [])
            vals = [fr.get("metrics", {}).get("vmaf") for fr in frames if "metrics" in fr and "vmaf" in fr.get("metrics", {})]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                raise RuntimeError("VMAF JSON missing values")
            return sum(vals) / len(vals)


class H265Compressor:
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.start_time = None

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        config_path = Path(__file__).parent / config_file
        default_config = {
            "algorithm_name": "H.265 High Efficiency",
            "parameters": {
                "preset": "slow",
                "crf": 34,
                "profile": "main",
                "level": "4.1",
                "tune": "psnr",
                "threads": 0,
                "tile_columns": 2,
                "tile_rows": 1
            },
            "audio": {
                "codec": "aac",
                "bitrate": "192k",
                "sample_rate": 48000
            },
            "use_hwaccel": False,
            # optimization_params used by golden_search (but scoring is custom now)
            "optimization_params": {
                "crf_min": 20,
                "crf_max": 40,
                "vmaf_threshold": 85,
                "w_c": 0.8,
                "w_vmaf": 0.2,
                "soft_threshold_margin": 5.0
            }
        }
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded = json.load(f)
                default_config.update(loaded)
            except Exception as e:
                print(f"Warning: failed to load config {config_file}: {e}")
        return default_config

    def _validate_input(self, input_video: str) -> bool:
        if not os.path.exists(input_video):
            print(f"Error: Input video file '{input_video}' does not exist!")
            return False
        try:
            with open(input_video, 'rb') as f:
                f.read(1024)
        except IOError:
            print(f"Error: Cannot read input video file '{input_video}'!")
            return False
        return True

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except Exception:
            pass
        return {}

    def _choose_encoder_for_extension(self, ext: str, detected_codec: Optional[str] = None) -> List[str]:
        """
        Choose encoder priorities by extension. For .avi we ensure libx264 (H.264) is chosen.
        detected_codec handling for VP/AV1 remains.
        """
        ext = ext.lstrip('.').lower()
        mapping = {
            'webm': ['libvpx-vp9', 'libaom-av1', 'libvpx-vp8'],
            'mkv': ['libx265', 'libx264'],
            'mp4': ['libx265', 'libx264'],
            'mov': ['libx265', 'libx264'],
            # Important: use H.264 when .avi (user requested)
            'avi': ['libx264', 'mpeg4']
        }
        candidates = mapping.get(ext, ['libx265', 'libx264'])
        # If the detected codec is a specialized codec prefer corresponding encoders
        if detected_codec in ('vp9', 'vp8', 'av1'):
            if detected_codec == 'vp9':
                return ['libvpx-vp9']
            if detected_codec == 'vp8':
                return ['libvpx-vp8']
            if detected_codec == 'av1':
                return ['libaom-av1']
        return [candidates[0]]

    def _detect_stream_codecs(self, info: Dict[str, Any]) -> Dict[str, Optional[str]]:
        ret = {'video': None, 'audio': None}
        try:
            for s in info.get('streams', []):
                if s.get('codec_type') == 'video' and not ret['video']:
                    ret['video'] = s.get('codec_name', '').lower()
                if s.get('codec_type') == 'audio' and not ret['audio']:
                    ret['audio'] = s.get('codec_name', '').lower()
        except Exception:
            pass
        return ret

    def _get_primary_video_stream(self, info: Dict[str, Any]) -> Dict[str, Any]:
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                return s
        return {}

    def _build_x265_params(self, params: Dict[str, Any]) -> str:
        x265_params = []
        tile_cols = params.get('tile_columns', 2)
        tile_rows = params.get('tile_rows', 1)
        if tile_cols > 0 and tile_rows > 0:
            x265_params.append(f"tiles={tile_cols}x{tile_rows}")
        x265_params.extend([
            "rc-lookahead=40",
            "bframes=8",
            "b-adapt=2",
            "ref=5",
            "me=hex",
            "subme=7",
            "rd=4"
        ])
        return ":".join(x265_params)

    def _build_ffmpeg_command(self, input_video: str, output_video: str) -> List[str]:
        params = self.config.get('parameters', {})
        audio_params = self.config.get('audio', {})

        info = self._get_video_info(input_video)
        streams = self._detect_stream_codecs(info)
        video_stream = self._get_primary_video_stream(info)

        input_ext = Path(input_video).suffix.lstrip('.').lower()
        chosen_encoders = self._choose_encoder_for_extension(input_ext, detected_codec=streams.get('video'))
        chosen_video_encoder = chosen_encoders[0]

        # ASSUMPTION: all inputs are 8-bit yuv420p per user instruction
        pix_fmt = 'yuv420p'

        # Decide audio flags
        audio_flags = []
        if input_ext == 'webm':
            if streams.get('audio') == 'opus':
                audio_flags = ['-c:a', 'copy']
            else:
                audio_flags = ['-c:a', 'libopus', '-b:a', audio_params.get('bitrate', '128k')]
        else:
            if streams.get('audio') == 'opus':
                audio_flags = ['-c:a', 'aac', '-b:a', audio_params.get('bitrate', '192k'), '-ar', '48000', '-ac', '2']
            else:
                audio_flags = ['-c:a', audio_params.get('codec', 'aac'), '-b:a', audio_params.get('bitrate', '192k'), '-ar', str(audio_params.get('sample_rate', 48000))]

        # Encoder-specific extra params
        video_encode_flags: List[str] = []
        if chosen_video_encoder == 'libx265':
            video_encode_flags = ['-c:v', 'libx265', '-preset', str(params.get('preset', 'slow')), '-crf', str(params.get('crf', 22)), '-profile:v', params.get('profile', 'main'), '-level', params.get('level', '4.1'), '-x265-params', self._build_x265_params(params)]
        elif chosen_video_encoder == 'libx264':
            # Provide sane defaults for libx264
            video_encode_flags = ['-c:v', 'libx264', '-preset', str(params.get('preset', 'slow')), '-crf', str(params.get('crf', 22)), '-profile:v', params.get('profile', 'high')]
        elif chosen_video_encoder == 'libvpx-vp9':
            video_encode_flags = ['-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', str(params.get('crf', 22)), '-threads', str(params.get('threads', 0)), '-tile-columns', str(params.get('tile_columns', 2)), '-g', '240', '-aq-mode', '0']
        elif chosen_video_encoder == 'libvpx-vp8':
            video_encode_flags = ['-c:v', 'libvpx', '-b:v', '0', '-crf', str(params.get('crf', 22))]
        elif chosen_video_encoder == 'libaom-av1':
            video_encode_flags = ['-c:v', 'libaom-av1', '-crf', str(params.get('crf', 22)), '-b:v', '0', '-cpu-used', '2']
        else:
            video_encode_flags = ['-c:v', chosen_video_encoder]

        # Probing flags
        probe_flags = ['-probesize', '50M', '-analyzeduration', '50M']
        hwaccel_flags = []
        if self.config.get('use_hwaccel'):
            hwaccel_flags = ['-hwaccel', 'auto']

        cmd = ['ffmpeg', '-y'] + probe_flags + hwaccel_flags + ['-i', input_video] + video_encode_flags + ['-pix_fmt', pix_fmt] + audio_flags + [output_video]

        tune = params.get('tune', 'none')
        if tune and tune != 'none' and '-x265-params' in ' '.join(video_encode_flags):
            cmd.extend(['-tune', tune])

        return cmd

    def _compute_vmaf(self, ref_video: str, dist_video: str) -> float:
        """
        Primary VMAF computation entry point: attempt fast sampled aligned VMAF,
        if that fails fallback to robust JSON-based ffmpeg/libvmaf pipeline.
        """
        print(f"[VMAF] Starting fast aligned VMAF calculation...")
        try:
            score = vmaf_mean_aligned_fast(ref_video, dist_video)
            print(f"[VMAF] Fast VMAF result: {score:.3f}")
            return score
        except Exception as e:
            print(f"[VMAF] Fast VMAF failed: {e}. Falling back to robust method...")
            return self._compute_vmaf_fallback(ref_video, dist_video)

    def _compute_vmaf_fallback(self, ref_video: str, dist_video: str) -> float:
        """Robust JSON-based ffmpeg/libvmaf pipeline (previous default)."""
        print(f"[VMAF] Starting VMAF fallback calculation...")
        try:
            ref_info = self._get_video_info(ref_video)
            ref_stream = self._get_primary_video_stream(ref_info)
            if not ref_stream:
                print("[VMAF] Could not read reference stream info; skipping VMAF")
                return 0.0

            w = int(ref_stream.get('width', 0) or 0)
            h = int(ref_stream.get('height', 0) or 0)
            rfr = ref_stream.get('r_frame_rate') or ref_stream.get('avg_frame_rate') or '0/1'
            try:
                fps = float(Fraction(rfr))
            except Exception:
                fps = 0.0

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as vmaf_log:
                log_path = vmaf_log.name

            scale_filter = ''
            if w > 0 and h > 0:
                scale_filter = f"scale={w}:{h}:flags=bicubic,"

            fps_filter = ''
            if fps > 0:
                fps_filter = f"fps=fps={fps},"

            # Force format to yuv420p for VMAF computation (inputs are 8-bit)
            lavfi = (
                f"[0:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[ref];"
                f"[1:v]setpts=PTS-STARTPTS,{scale_filter}{fps_filter}format=yuv420p[dist];"
                f"[ref][dist]libvmaf=log_fmt=json:log_path={log_path}"
            )

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i", str(Path(ref_video).resolve()),
                "-i", str(Path(dist_video).resolve()),
                "-lavfi", lavfi,
                "-f", "null", "-"
            ]

            print(f"[VMAF] Running fallback ffmpeg/libvmaf (command suppressed)...")
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1200,
                check=False
            )

            if Path(log_path).exists() and Path(log_path).stat().st_size > 10:
                try:
                    with open(log_path, "r") as f:
                        vmaf_data = json.load(f)
                    frames = vmaf_data.get('frames', [])
                    if frames:
                        vmaf_scores = [float(frame.get('metrics', {}).get('vmaf', 0)) for frame in frames if 'vmaf' in frame.get('metrics', {})]
                        if vmaf_scores:
                            score = sum(vmaf_scores) / len(vmaf_scores)
                            print(f"[VMAF] Parsed JSON VMAF (fallback): {score:.3f} (using {log_path})")
                            return score
                except Exception as e:
                    print(f"[VMAF] JSON parse error in fallback: {e}")

            print(f"[VMAF] ffmpeg exitcode={proc.returncode}")
            print(f"[VMAF] ffmpeg stderr (last 2000 chars):\n{(proc.stderr or '')[-2000:]}")
            print(f"[VMAF] VMAF fallback calculation failed or returned no frames.")
            return 0.0
        finally:
            try:
                if 'log_path' in locals() and Path(log_path).exists():
                    Path(log_path).unlink()
            except Exception:
                pass

    def _score_crf(self, input_video: str, crf: int) -> Tuple[float, Optional[str]]:
        """
        Encode with CRF, compute the new golden-search score:
            compression = initial_size / final_size
            score = 0.7 * compression + 0.3 * vmaf
        Higher score is better.
        Returns (score, output_file_path) or (-inf, None) on failure.
        """
        params = dict(self.config.get("parameters", {}))
        params["crf"] = crf

        tmp_output = Path(tempfile.gettempdir()) / f"tmp_crf{crf}{Path(input_video).suffix}"
        if tmp_output.exists():
            try:
                tmp_output.unlink()
            except Exception:
                pass

        saved_config = dict(self.config)
        try:
            self.config = dict(self.config)
            self.config['parameters'] = params

            cmd = self._build_ffmpeg_command(input_video, str(tmp_output))
            print(f"[SCORE] Encoding with CRF={crf} -> {' '.join(cmd[:6])} ...")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                print(f"[SCORE] ffmpeg failed for CRF={crf} rc={proc.returncode}")
                print((proc.stderr or '')[-1000:])
                return -1e9, None

            if not tmp_output.exists():
                return -1e9, None

            orig_size = os.path.getsize(input_video)
            comp_size = os.path.getsize(tmp_output)

            # Per user request: compression = initial_size / final_size
            if comp_size > 0:
                compression = orig_size / comp_size
            else:
                compression = 0.0

            vmaf = self._compute_vmaf(input_video, str(tmp_output))

            # New scoring formula: score = 0.7 * compression + 0.3 * vmaf
            score = 0.7 * compression + 0.3 * vmaf

            print(f"[TEST] CRF={crf}, Orig={orig_size} bytes, Comp={comp_size} bytes, Compression={compression:.6f}, VMAF={vmaf:.2f}, Score={score:.6f}")
            return score, str(tmp_output)
        except subprocess.TimeoutExpired:
            print(f"[SCORE] CRF={crf} encoding timed out")
            return -1e9, None
        except Exception as e:
            print(f"[SCORE] Exception for CRF={crf}: {e}")
            return -1e9, None
        finally:
            self.config = saved_config

    def golden_search(self, input_video: str) -> Tuple[int, Optional[str], float]:
        opt_params = self.config.get("optimization_params", {})
        a = opt_params.get("crf_min", 20)
        b = opt_params.get("crf_max", 40)

        phi = (math.sqrt(5) - 1) / 2
        tol = 1  # stop when CRF interval <= 1

        x1 = int(b - phi * (b - a))
        x2 = int(a + phi * (b - a))

        print(f"[GOLDEN] Starting parallel scoring for CRF {x1} and {x2}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_map = {executor.submit(self._score_crf, input_video, x): x for x in (x1, x2)}
            results_map = {}
            for future in as_completed(future_map):
                crf = future_map[future]
                score, outfile = future.result()
                results_map[crf] = (score, outfile)

        f1, out1 = results_map.get(x1, (-1e9, None))
        f2, out2 = results_map.get(x2, (-1e9, None))

        while abs(b - a) > tol:
            print(f"[GOLDEN] Interval: [{a}, {b}] | CRFs: {x1}, {x2}")
            if f1 > f2:
                # drop right
                b, f2, out2, x2 = x2, f1, out1, x1
                x1 = int(b - phi * (b - a))
                print(f"[GOLDEN] Scoring CRF {x1}")
                f1, out1 = self._score_crf(input_video, x1)
            else:
                # drop left
                a, f1, out1, x1 = x1, f2, out2, x2
                x2 = int(a + phi * (b - a))
                print(f"[GOLDEN] Scoring CRF {x2}")
                f2, out2 = self._score_crf(input_video, x2)

        if f1 > f2:
            return x1, out1, f1
        else:
            return x2, out2, f2

    def compress(self, input_video: str, output_video: str) -> bool:
        self.start_time = time.time()
        print(f"[STEP] Starting compression (Linux-ready)...")
        print(f"[STEP] Input: {input_video}")
        print(f"[STEP] Output: {output_video}")

        if not self._validate_input(input_video):
            print(f"[STEP] Input validation failed.")
            return False

        input_ext = Path(input_video).suffix
        if not output_video.endswith(input_ext):
            output_video = str(Path(output_video).with_suffix(input_ext))
            print(f"[STEP] Output extension forced to match input: {output_video}")

        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        print(f"[STEP] Output directory ensured.")

        info = self._get_video_info(input_video)
        if info:
            fmt = info.get('format', {})
            try:
                print(f"[STEP] Input size {int(fmt.get('size',0))/(1024*1024):.1f} MB, duration {float(fmt.get('duration',0)):.1f}s")
            except Exception:
                pass

        cmd = self._build_ffmpeg_command(input_video, output_video)
        print(f"[STEP] FFmpeg command: {' '.join(cmd)}")

        try:
            params = self.config.get('parameters', {})
            crf_value = params.get('crf', 34)
            preset_value = params.get('preset', 'slow')
            print(f"[STEP] Preset: {preset_value}, CRF: {crf_value}")

            print(f"[STEP] Running FFmpeg compression...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                end_time = time.time()
                duration = end_time - self.start_time
                if os.path.exists(output_video):
                    out_mb = os.path.getsize(output_video)/(1024*1024)
                    print(f"[STEP] Compression success: {out_mb:.1f} MB in {duration:.1f}s")
                else:
                    print(f"[STEP] Compression completed in {duration:.1f}s (no output file detected)")
                return True
            else:
                print(f"[STEP] FFmpeg failed (rc={result.returncode})")
                print((result.stderr or '')[-2000:])
                return False
        except subprocess.TimeoutExpired:
            print("[STEP] Compression timed out")
            return False
        except Exception as e:
            print(f"[STEP] Compression exception: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Linux-ready H.265/HEVC Video Compressor with Golden Search (modified)')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file (extension preserved to match input)')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()

    compressor = H265Compressor(args.config)
    best_crf, best_file, best_sf = compressor.golden_search(args.input)

    if not best_file:
        print("[RESULT] No valid output produced by golden search.")
        sys.exit(2)

    final_output = args.output
    input_ext = Path(args.input).suffix
    if not final_output.endswith(input_ext):
        final_output = str(Path(final_output).with_suffix(input_ext))

    shutil.move(best_file, final_output)

    print(f"\n[RESULT] Best CRF={best_crf}, Score={best_sf:.6f}")
    print(f"[OUTPUT] Final compressed video: {final_output}")


if __name__ == '__main__':
    main()
