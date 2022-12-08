
run:
        docker build -t dockerimdb/workspace/spark_docker_vlerick/docker_exercises/dockerfile_imdbnotebook/dockerfile_imbd .
        docker run --name my-jupyter-kotlin \
        -v /workspace/spark_docker_vlerick:/workdir:rw \
        -p 8080
        dockerimdb/workspace/spark_docker_vlerick/docker_exercises/dockerfile_imdbnotebook/dockerfile_imbd 

		