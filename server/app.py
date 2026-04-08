import uvicorn
from server_main import app


def main():
    uvicorn.run(
        "server_main:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )


if __name__ == "__main__":
    main()