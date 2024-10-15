# Image Classification with Vision Transformer and Swin Transformer
This project implements and compares two state-of-the-art transformer-based models for image classification: Vision Transformer (ViT) and Swin Transformer. The dataset used is a flower classification dataset containing 13 classes. The models are trained and evaluated to understand their performance differences and the advantages each brings to the table.

## Dataset Overview:
- **Dataset**: Flower Classification Dataset
- **Number of Classes**: 13
- **Image Size**: Each image is resized to 224x224 pixels before being fed into the models.

## Task Objective:
The goal of this project is to classify images of flowers into 13 distinct categories using both Vision Transformer and Swin Transformer models. By comparing their architectures and performances, we can explore their strengths and limitations on the same task.

---

## 1. Vision Transformer (ViT)
The Vision Transformer model treats an image as a sequence of patches and applies transformer architecture to model the relationships between patches. Here's a breakdown of the setup and configurations used for training this model:

### Configuration Details
- **Image Size**: 224x224
  - Each input image is resized to 224x224 pixels during preprocessing.
- **Patch Size**: 8x8
  - The image is divided into non-overlapping patches of size 8x8, resulting in (224/8)² = 784 patches per image.
- **Model Architecture**:
  - A single **Multi-Head Self-Attention Block** is used.
  - Each attention block is followed by an **MLPBlock** to transform the output of the attention mechanism.
  - **Positional Embedding** is added to the patches to retain spatial information.
  - **Transformer Encoder Block** includes Layer Normalization, followed by multi-head self-attention, followed by feed-forward neural networks (MLPBlock).
- **Batch Size**: 16
  - The model is trained with a batch size of 16 images at a time.
- **Optimizer**: Adam
  - Adaptive learning rate optimization is used with the Adam optimizer.
  - **Learning Rate**: 0.0001
  - **Weight Decay**: N/A (not used in this model)
- **Loss Function**: Cross Entropy Loss
  - Cross Entropy Loss is used as the loss function for classification.
- **Epochs**: 50
  - The model is trained for 50 epochs.
- **Device**: CUDA (GPU acceleration)

### DataLoader
- **Image Preprocessing**: 
  - Each image is resized to 224x224 pixels before being divided into patches.
- **Dataloader**:
  - **Batch Size**: 16
  - A dataloader is created to fetch batches of images and their corresponding labels during training.
    
### Model Training and Optimization
The training of the Vision Transformer involves passing the sequence of patches through the transformer encoder blocks. Each patch is embedded, positional encodings are added, and the self-attention mechanism allows the model to learn the relationships between patches. The output is passed through a classification head, and the model is optimized using the **Adam optimizer**.
- **Optimization**:
  - **Learning Rate**: 0.0001
  - **Loss Function**: Cross Entropy Loss
  - **Epochs**: 50
- **Training Progress**: The model was trained for 50 epochs with a learning rate of 0.0001, and Cross Entropy Loss was used to optimize the classification accuracy.

---

## 2. Swin Transformer
The Swin Transformer improves upon ViT by introducing a hierarchical structure and local window-based attention. This design enables the Swin Transformer to handle images more effectively, especially for capturing both local and global patterns.

### Configuration Details
- **Image Size**: 224x224
  - Each input image is resized to 224x224 pixels.
- **Window Size**: 4x4
  - Attention is applied within 4x4 windows. The local attention mechanism helps reduce computational complexity while focusing on smaller regions of the image.
- **Patch Merging**:
  - Swin Transformer introduces **Patch Merging** to progressively downsample the feature map and extract hierarchical features.
  - After merging, patches are combined to reduce the size of the feature map, allowing the model to capture larger and more global patterns.
  - **Number of Patches**: 784 (initially before merging)
  - **Number of Channels**: 192 (features per patch after merging)
- **Batch Size**: 
  - The model is trained with a **batch size of 1** during training (due to computational constraints), but dataloaders are configured with a batch size of 16.
- **Optimizer**: AdamW
  - **Learning Rate**: 0.0001
  - **Weight Decay**: 0.003
- **Loss Function**: Cross Entropy Loss
- **Epochs**: 50
- **Device**: L4 GPU (CUDA)

### DataLoader
- **Image Preprocessing**:
  - All images are resized to 224x224 pixels.
- **Dataloader**:
  - **Batch Size**: 16 (for creating dataloaders).

### Model Training and Optimization
The Swin Transformer uses a **hierarchical design** with **shifted windows** to progressively downsample the input image. This allows the model to capture both local and global features more efficiently than the Vision Transformer. Patch merging further enables the model to handle larger images with better computational efficiency.
- **Optimization**:
  - **Learning Rate**: 0.0001
  - **Weight Decay**: 0.003
  - **Loss Function**: Cross Entropy Loss
  - **Epochs**: 50
- **Training Progress**: Trained for 50 epochs with the AdamW optimizer and Cross Entropy Loss.

---

## Results and Inference:

### Performance Comparison
- The **Swin Transformer** model achieved higher classification accuracy compared to the **Vision Transformer** on the flower classification dataset.
- The performance boost of Swin Transformer is particularly noticeable in its ability to capture fine details in complex flower images.

### Key Insights:
1. **Hierarchical Feature Extraction**:
   - Swin Transformer’s hierarchical structure allows it to capture both local and global features more efficiently than the Vision Transformer. By progressively merging patches, the model builds a more informative representation of the image at different scales.
   
2. **Localized Attention with Windows**:
   - Swin Transformer uses window-based attention, meaning attention is applied only to a subset of patches in localized regions, reducing computational complexity and improving performance by focusing on fine details. The **shifted window mechanism** helps in capturing cross-window information, providing a balance between local and global representation.
   
3. **Patch Merging**:
   - The Swin Transformer’s ability to downsample the input via patch merging allows it to focus on larger parts of the image, capturing both detailed textures and overall structure, which is especially important in distinguishing flower species.

4. **Efficiency**:
   - The Vision Transformer applies attention globally to all patches, which can become computationally expensive, especially for large images or tasks that involve complex patterns. The Swin Transformer, by contrast, is more efficient due to its window-based attention mechanism and patch merging.
   
---

## Conclusion
Both models demonstrate strong performance on the flower classification task, but the **Swin Transformer** proves to be more effective due to its efficient **hierarchical feature extraction**, **window-based attention**, and **patch merging** techniques. These improvements allow Swin to generalize better to complex image classification tasks, making it more suitable for real-world applications involving high-resolution or intricate patterns.

## Contributors:  
[Pravalika Arunkumar](https://github.com/pravalikaarunkumar)
