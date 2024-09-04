pipeline {
    agent any
    parameters {
        booleanParam(name: 'RUN_SPLIT_DATA', defaultValue: true, description: 'Run Split Data Stage')
        booleanParam(name: 'RUN_TRAIN', defaultValue: true, description: 'Run Train Stage')
        booleanParam(name: 'RUN_PREDICT', defaultValue: true, description: 'Run Predict Stage')
    }
    stages {
        stage('Clone Repository') {
            steps {
                git url: 'https://github.com/fredy737/data-science.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    {
                        sh 'docker build -t reddit_bert .'
                    }
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    {
                        sh 'kubectl apply -f src/kubernetes/deployment.yaml'
                        sh 'kubectl apply -f src/kubernetes/service.yaml'
                    }
                }
            }
        }
    }
}
