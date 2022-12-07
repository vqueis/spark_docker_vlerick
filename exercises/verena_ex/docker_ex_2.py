docker image pull devopsdockeruh/simple-web-service:ubuntu 
docker ps
docker exec -it musing_swanson /bin/bash
tail -f ./text.log
control c
exit

#or

docker pull devopsdockeruh/simple-web-service:ubuntu
docker ps
exec -it musing_swanson /bin/bash

#to stop the docker from running
#first type docker ps to see if it is running, then write
docker stop musing_swanson