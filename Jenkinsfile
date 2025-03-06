pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = "wahidimahrukh/shoe-price-predictions"
        DOCKER_HUB_CREDENTIALS = "docker-hub-credentials"
    }

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'master', url: 'https://github.com/Mahrukh272004/shoe-price-predictions.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t $DOCKER_HUB_REPO:latest ."
                }
            }
        }

        stage('Push Image to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_HUB_CREDENTIALS) {
                        sh "docker push $DOCKER_HUB_REPO:latest"
                    }
                }
            }
        }

        stage('Deploy Container') {
            steps {
                script {
                    sh "docker run -d --name shoe-app -p 5000:5000 $DOCKER_HUB_REPO:latest"
                }
            }
        }
    }
}
