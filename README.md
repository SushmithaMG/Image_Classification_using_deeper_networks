In this project, a hybrid convolutional neural network is constructed by combining pretrained ResNet18 and GoogleNet models, to perform image classification.
To enable parameter-efficient fine-tuning, selected convolutional layers are augmented with Low-Rank Adaptation (LoRA) modules. The model is evaluated on the MNIST and CIFAR datasets, with ablation studies involving LoRA, to assess the contributions of individual components. Re-
sults demonstrate that the LoRA-augmented hybrid model achieves high classification accuracy with minimal additional training overhead, underscoring its potential for scalable and efficient deployment in
resource-constrained settings.
