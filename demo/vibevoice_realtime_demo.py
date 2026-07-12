import argparse, os, uvicorn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--model_path", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mpx", "mps"])
    p.add_argument("--reload", action="store_true", help="Reload the model or not")
    args = p.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MODEL_DEVICE"] = args.device

    uvicorn.run("web.app:app", host="0.0.0.0", port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
