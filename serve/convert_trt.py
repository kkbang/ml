"""
convert_trt.py — ONNX(inputs_embeds 입력) → TensorRT 변환
실행: python convert_trt.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

ROOT = Path(os.getenv('PROJECT_ROOT'))

sys.path.insert(0, str(ROOT / 'core'))
sys.path.append(str(ROOT / 'parser'))

import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from dataset import TOTAL_LENGTH

ONNX_PATH = 'graphcodebert_encoder.onnx'
TRT_PATH  = 'graphcodebert_encoder.trt'
MAX_BATCH = 64
FP16_MODE = True
LOG_LEVEL = trt.Logger.WARNING
L = TOTAL_LENGTH   # 320
D = 768


def build_trt_engine():
    print("=== TensorRT 엔진 빌드 ===\n")
    logger  = trt.Logger(LOG_LEVEL)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"ONNX 파싱 중: {ONNX_PATH}")
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  오류: {parser.get_error(i)}")
            raise RuntimeError("ONNX 파싱 실패")
    print("  ✓ 파싱 완료")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if FP16_MODE:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 모드 활성화")

    # Dynamic shape: inputs_embeds(B,L,D), attention_mask(B,L), position_ids(B,L)
    profile = builder.create_optimization_profile()
    profile.set_shape('inputs_embeds',
        min=(1, L, D), opt=(16, L, D), max=(MAX_BATCH, L, D))
    profile.set_shape('attention_mask',
        min=(1, L),    opt=(16, L),    max=(MAX_BATCH, L))
    profile.set_shape('position_ids',
        min=(1, L),    opt=(16, L),    max=(MAX_BATCH, L))
    config.add_optimization_profile(profile)

    print(f"\n엔진 빌드 중... (MAX_BATCH={MAX_BATCH}, FP16={FP16_MODE})")
    t0  = time.time()
    eng = builder.build_serialized_network(network, config)
    if eng is None:
        raise RuntimeError("TRT 빌드 실패")
    print(f"  ✓ 빌드 완료 ({time.time()-t0:.1f}초)")

    with open(TRT_PATH, 'wb') as f:
        f.write(eng)
    print(f"  → {TRT_PATH} 저장 완료")


def benchmark_trt():
    print(f"\n=== TensorRT 벤치마크 ===")
    logger  = trt.Logger(LOG_LEVEL)
    runtime = trt.Runtime(logger)
    with open(TRT_PATH, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    for B in [1, 8, 16, 32, 64]:
        embeds = np.random.randn(B, L, D).astype(np.float32)
        attn   = np.ones((B, L), dtype=np.int64)
        pos    = np.ones((B, L), dtype=np.int64)
        out    = np.zeros((B, D), dtype=np.float32)

        d_emb  = cuda.mem_alloc(embeds.nbytes)
        d_attn = cuda.mem_alloc(attn.nbytes)
        d_pos  = cuda.mem_alloc(pos.nbytes)
        d_out  = cuda.mem_alloc(out.nbytes)
        stream = cuda.Stream()

        context.set_input_shape('inputs_embeds',  (B, L, D))
        context.set_input_shape('attention_mask', (B, L))
        context.set_input_shape('position_ids',   (B, L))
        context.set_tensor_address('inputs_embeds',  int(d_emb))
        context.set_tensor_address('attention_mask', int(d_attn))
        context.set_tensor_address('position_ids',   int(d_pos))
        context.set_tensor_address('cls_output',     int(d_out))

        # warm-up
        for _ in range(5):
            cuda.memcpy_htod_async(d_emb, embeds, stream)
            cuda.memcpy_htod_async(d_attn, attn,  stream)
            cuda.memcpy_htod_async(d_pos,  pos,   stream)
            context.execute_async_v3(stream.handle)
            stream.synchronize()

        # NaN 체크
        cuda.memcpy_dtoh_async(out, d_out, stream); stream.synchronize()
        nan_flag = np.isnan(out).any()

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            cuda.memcpy_htod_async(d_emb, embeds, stream)
            cuda.memcpy_htod_async(d_attn, attn,  stream)
            cuda.memcpy_htod_async(d_pos,  pos,   stream)
            context.execute_async_v3(stream.handle)
            cuda.memcpy_dtoh_async(out, d_out, stream)
            stream.synchronize()
            times.append((time.perf_counter()-t0)*1000)

        avg_ms = np.mean(times)
        qps    = B / (avg_ms / 1000)
        nan_str = " ⚠️ NaN!" if nan_flag else ""
        print(f"  batch={B:2d} | {avg_ms:.1f}ms | {qps:.0f} QPS{nan_str}")

        d_emb.free(); d_attn.free(); d_pos.free(); d_out.free()


if __name__ == '__main__':
    build_trt_engine()
    benchmark_trt()
    print(f"\n완료. 다음 단계: uvicorn embed_server:app ...")
