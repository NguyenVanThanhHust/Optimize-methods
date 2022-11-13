# Install instruction
- Build docker
```
docker build -t deepstream_60_python -f Dockerfile .
```

- Start dockker 
```
docker run -it --rm --gpus=all --name ds_cnt --shm-size=8G --memory=32G  --volume="$PWD:/workspace" deepstream_60_python:latest
```

```
docker exec -it ds_cnt /bin/bash 
```

- Inside docker, build gst_python as in link [gstreamer-python](https://github.com/jackersson/gstreamer-python)

https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings

https://forums.developer.nvidia.com/t/compiling-deepstream-6-0-python-bindings-pybind11-h-no-such-file-or-directory/195232/3