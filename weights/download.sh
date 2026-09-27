set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt"
if [ ! -f yolo11m.pt ]; then
  echo "Downloading yolo11m.pt ..."
  curl -L --fail -o yolo11m.pt "$URL"
fi
echo "OK: $(pwd)/yolo11m.pt"
