pipeline {
    agent any

    environment {
        DOCKER_HUB_USERNAME = credentials('docker-hub-credentials') // Fetch Docker Hub credentials
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t $DOCKER_HUB_USERNAME/shoe-price-predictions:latest ."
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                script {
                    sh "echo ${DOCKER_HUB_PASSWORD} | docker login -u ${DOCKER_HUB_USERNAME} --password-stdin"
                    sh "docker push $DOCKER_HUB_USERNAME/shoe-price-predictions:latest"
                }
            }
        }
    }
}
