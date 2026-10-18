import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local Personal AI Workbench")
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--data",default="data")
    args = parser.parse_args()
    os.environ["PERSONAL_AI_DATA"] = os.path.abspath(args.data)
    uvicorn.run("personal_ai.api:create_app",factory=True,host="127.0.0.1",port=args.port)


if __name__ == "__main__":
    main()
