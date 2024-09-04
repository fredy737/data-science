pipeline {
    agent {
        kubernetes {
            label 'reddit-bert-pipeline'
        }
    }
    parameters {
        booleanParam(name: 'RUN_SPLIT_DATA', defaultValue: true, description: 'Run Split Data Stage')
        booleanParam(name: 'RUN_TRAIN', defaultValue: true, description: 'Run Train Stage')
        booleanParam(name: 'RUN_PREDICT', defaultValue: true, description: 'Run Predict Stage')
    }
    environment {
        STAGES_TO_RUN = ''
    }
    stages {
        stage('Determine Stages') {
            steps {
                script {
                    def stages_to_run = []
                    if (params.RUN_SPLIT_DATA) {
                        stages_to_run.add('split_data')
                    }
                    if (params.RUN_TRAIN) {
                        stages_to_run.add('train')
                    }
                    if (params.RUN_PREDICT) {
                        stages_to_run.add('predict')
                    }
                    env.STAGES_TO_RUN = stages_to_run.join(' ')
                }
            }
        }
        stage('Run Pipeline') {
            when {
                expression {
                    return env.STAGES_TO_RUN != ''
                }
            }
            steps {
                script {
                    echo "Running pipeline stages: ${env.STAGES_TO_RUN}"
                    sh "kubectl set env deployment/pipeline-deployment STAGES=${env.STAGES_TO_RUN}"
                    sh "kubectl rollout restart deployment pipeline-deployment"
                }
            }
        }
    }
}
