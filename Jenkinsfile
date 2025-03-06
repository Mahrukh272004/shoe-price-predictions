pipeline {
    agent any

    environment {
        // Docker Hub credentials (stored in Jenkins credentials)
        DOCKER_HUB_USERNAME = credentials('DOCKER_HUB_USERNAME')
        DOCKER_HUB_TOKEN = credentials('DOCKER_HUB_TOKEN')
        DOCKER_IMAGE_NAME = "wahidimahrukh/shoe-price-predictions"
        DOCKER_IMAGE_TAG = "latest"
    }

    stages {
        // Stage 1: Build Docker Image
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image..."
                    docker.build("${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}")
                }
            }
        }

        // Stage 2: Push Docker Image to Docker Hub
        stage('Push Docker Image to Docker Hub') {
            steps {
                script {
                    echo "Logging into Docker Hub..."
                    sh "echo ${DOCKER_HUB_TOKEN} | docker login -u ${DOCKER_HUB_USERNAME} --password-stdin"

                    echo "Pushing Docker image to Docker Hub..."
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}").push()
                    }
                }
            }
        }

        // Stage 3: Deploy Application
        stage('Deploy Application') {
            steps {
                script {
                    echo "Deploying application..."
                    // Example: Use docker-compose to deploy the application
                    sh "docker-compose down" // Stop existing containers
                    sh "docker-compose up -d" // Start new containers
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded! Application deployed successfully."
        }
        failure {
            echo "Pipeline failed. Check the logs for errors."
        }
    }
}