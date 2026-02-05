import argparse
import tensorrt as trt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    print("TensorRT version:", trt.__version__)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    with open(args.engine, "rb") as f, trt.Runtime(logger) as runtime:
        engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("Failed to deserialize engine (engine is None)")

    print("OK: engine deserialized")
    print("num optimization profiles:", engine.num_optimization_profiles)
    print("num I/O tensors:", engine.num_io_tensors)

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"  tensor[{i}] name={name} mode={mode} dtype={dtype} shape={shape}")

        # show per-profile shapes for this tensor if input
        if mode == trt.TensorIOMode.INPUT and engine.num_optimization_profiles > 0:
            for p in range(engine.num_optimization_profiles):
                mn, opt, mx = engine.get_tensor_profile_shape(name, p)
                print(f"    profile[{p}] min={mn} opt={opt} max={mx}")

    print("\nCreating execution context...")
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("create_execution_context() returned None")
    print("OK: execution context created")

if __name__ == "__main__":
    main()
