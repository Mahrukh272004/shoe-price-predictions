pipeline {
    agent any

    environment {
        DOCKER_HUB_CREDENTIALS = credentials('docker-hub-credentials')
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("wahidimahrukh/shoe-price-predictions:${env.BUILD_ID}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("wahidimahrukh/shoe-price-predictions:${env.BUILD_ID}").push()
                    }
                }
            }
        }
    }
}