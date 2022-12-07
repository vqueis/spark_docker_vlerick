docker image pull devopsdockeruh/simple-web-service:ubuntu
docker ps
docker exec -it musing_swanson /bin/bash
tail -f ./text.log
control c
exit

or

docker pull devopsdockeruh/simple-web-service:ubuntu
docker ps
exec -it musing_swanson /bin/bash
