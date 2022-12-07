docker run \
    --rm -i \
    -e PGID=$(id -g) \
    -e PUID=$(id -u) \
    -v /workspace/spark_docker_vlerick:/workdir:rw \
    ghcr.io/mikenye/docker-youtube-dl:latest https://www.youtube.com/watch?v=psmZRfiXYnE
