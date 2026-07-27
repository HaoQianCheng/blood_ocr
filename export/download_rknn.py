"""下载 rknn-toolkit2 2.3.2 whl(WSL 里 curl/wget 缺失, 用 python urllib)。

在 WSL 里运行:
    python3 /mnt/c/Users/qianc/Projects/blood_ocr/export/download_rknn.py
"""
import sys
import urllib.request

URL = (
    "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/"
    "rknn-toolkit2/packages/x86_64/"
    "rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
)
OUT = "/tmp/rknn_toolkit2.whl"


def main():
    print(f"下载: {URL}")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            print(f"大小: {total} bytes ({total / 1024 / 1024:.1f} MB)")
            downloaded = 0
            with open(OUT, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\r  {downloaded / 1024 / 1024:.1f}"
                            f"/{total / 1024 / 1024:.1f} MB",
                            end="",
                            flush=True,
                        )
        print(f"\n完成: {OUT}")
    except Exception as e:
        print(f"\n失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
