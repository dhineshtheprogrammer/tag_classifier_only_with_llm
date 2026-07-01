from .CustomAOI_helper import InitAOI

llmModels = ["gpt-35-turbo", "gpt-4o","gpt-4o-mini", "gpt-5.1"]
deploymentNames = ["genAICoEDevelopmentAndTesting",
                   "genAICoEDevelopmentAndTestingGPT4oLLM",
                   "genAICoEDevelopmentAndTesting4oMini",
                   "genAICoEDevelopmentAndTestingGPT5v1"]
embModel = "text-embedding-ada-002"
embDeploymentName = "DevelopmentAndTestingEmbedder"

secure_models = InitAOI(__name__, "USERNAME", "PASSWORD", llmModels, deploymentNames, embModel, embDeploymentName)