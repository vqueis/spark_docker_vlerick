

### 6. Package this set of code in a Docker image that you must push to the AWS elastic container registry ###
### Links to an external site.(ECR) bearing the name 338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions ###
### and with a tag that starts with your first name. ###

FROM python:3.9.12
LABEL Verena <vqueis@gmail.com>:

# Install apt packages and clean up cached files
RUN apt-get update && apt-get install -y openjdk-11-jdk 

#RUN python3 version RUN pip3 --version 

WORKDIR /workspace/spark_docker_vlerick/docker_exercises/dockerfile_imdbnotebook 
COPY . /workspace/spark_docker_vlerick/docker_exercises/dockerfile_imdbnotebook 

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



# Install the AWS CLI and clean up tmp files
RUN wget https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -O ./awscliv2.zip && \
   unzip awscliv2.zip && \
   ./aws/install
   #krm -rf ./aws awscliv2.zip

#RUN pip install awscli
#RUN --mount=type=secret,id=aws,target=/root/.aws/credentials aws s3a cp s3a://338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions

# For vscode
EXPOSE 3000
# for spark
EXPOSE 4040

ENTRYPOINT [ "python3" ]
CMD [ "code.py"] 




# Build the Docker image using the Dockerfile
#$ docker build -t my_image .  (docker image build -t vq-image3 /workspace/spark_docker_vlerick/docker_exercises/dockerfile_imdbnotebook)

#docker run -e AWS_ACCESS_KEY_ID=xyz -e AWS_SECRET_ACCESS_KEY=aaa myimage         vul in xyz en aaa codes van outlook


# Tag the Docker image with a unique name
#$ docker tag vq-image2 338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions_verena

# Push the Docker image to ECR
#$ docker push 338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions_verena





