pipeline {
    agent any

    environment {
        S3_BUCKET = 'fredy-data'
    }

    stages {
        stage('Clone repository') {
            steps {
                // Clone Git repository
                git url: 'https://github.com/fredy737/data-science.git', branch: 'imp/reddit-comments-bert-2'
            }
        }
        stage('Set up Python environment') {
            steps {
                // Set up virtual environment and install dependencies
                sh 'python 3 -m venv venv'
                sh './venv/bin/pip install -r data-science/reddit_bert/requirements.txt'
            }
        }
        stage('prepare_data') {
            steps {
                script {
                    sh './venv/bin/python data-science/reddit_bert/commands/prepare_data.py'
                }
            }
        }
        stage('split_data') {
            steps {
                script {
                    sh './venv/bin/python data-science/reddit_bert/commands/split_data.py'
                }
            }
        }
        stage('train') {
            steps {
                script {
                    sh './venv/bin/python data-science/reddit_bert/commands/train.py'
                }
            }
        }
    }
    post {
        always {
            // Cleanup actions
            cleanWs()
        }
    }
}
