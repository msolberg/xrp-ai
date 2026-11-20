podman run --rm -it \
  --userns=keep-id:uid=1001,gid=0 \
  --device nvidia.com/gpu=all \
  --device /dev/ttyACM0 \
  --device /dev/snd \
  --device /dev/video0 \
  --group-add keep-groups \
  --security-opt label=disable \
  -v ./voices:/app/voices:Z \
  -v ./sounds:/app/sounds:Z \
  -v ./models:/app/models:Z \
  -v ./vosk-cache:/app/.cache/vosk:Z \
  xrp-ai:latest --loglevel INFO

