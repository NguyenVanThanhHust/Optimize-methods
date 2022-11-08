# Install instruction
- Build docker
```
nvidia-docker build -f deepstream_image -f Dockerfile .
```

- Start dockker 
```
docker run -it --gpus=all --name ds_cnt_ffmpeg --shm-size=8G --memory=32G  --volume="$PWD:/workspace" deepstream_60:latest
```

```
docker exec -it ds_cnt_ffmpeg /bin/bash 
```

- Inside docker, build gst_python as in link [gstreamer-python](https://github.com/jackersson/gstreamer-python)

