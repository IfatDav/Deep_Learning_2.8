# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:** Ifat Davidson  **ID:** 037377140  **Date:** 31.7.26
**Chosen option:** B (A · Olist MLP / B · Fashion-MNIST CNN / C · Olist Autoencoder)

---

## 1. Framing
Task and primary metric: Multi-class image classification of clothing items (Fashion-MNIST). The primary metrics are Validation Accuracy and Macro F1 Score.

Baseline you are comparing against: A simple linear model (e.g., Logistic Regression) operating on flattened 2D image data.

---

## 2. Results
| Model | Test metric | Params | Train time | Notes |
|---|---|---|---|---|
| simpler baseline | ~84.0% Accuracy| ~7,850| ~5 seconds| Requires flattening 2D spatial data, losing structural context.|
| neural net |~92.7% Accuracy |421,642 |~2-3 minutes |CNN preserves spatial relationships via convolutional filters. |

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Did DL win?** Did your neural net beat the simpler / cheaper baseline? By how much, and at what cost?
Yes, the Convolutional Neural Network beat the simpler baseline by a significant margin of approximately 8% in accuracy (improving from ~84% to ~92.7%). However, this improvement came at the cost of a massive increase in computational complexity (421k parameters versus 7.8k), substantially longer training times, and the loss of direct feature interpretability.
2. **Logits / loss.** Which loss did you use and why? What breaks if you apply softmax / sigmoid before it?
I used CrossEntropyLoss because it is the standard and most stable loss function for multi-class classification tasks in PyTorch. It mathematically combines LogSoftmax and NLLLoss into a single operation. Applying a Softmax or Sigmoid activation function to the outputs before passing them to this loss function disrupts this mathematical stability, leading to vanishing gradients and severely degraded optimization.
3. **Overfitting.** Read your learning curves. Where do train and val diverge, and what did you do about it?
By analyzing the learning curves, the training and validation losses begin to diverge around Epoch 8. At this point, the validation loss reaches its absolute minimum (0.2186) and begins to slightly climb, while the training loss continues its steady decline. To prevent the model from memorizing the training data, I utilized an Early Stopping mechanism with a patience of 3 epochs, which successfully halted the training process at Epoch 11 and restored the optimal weights.
4. **Learning rate.** Show what happens with a too-large and a too-small LR. Why is it the most important knob?
Applying a too-large learning rate (e.g., 1e-1) causes the loss to oscillate erratically or explode because the optimizer takes steps that are too aggressive, repeatedly overshooting the local minimum. Conversely, a too-small learning rate (e.g., 1e-5) causes the network to learn at a microscopic pace, failing to converge within a reasonable number of epochs. The learning rate is the most critical knob because it dictates the step size for gradient descent, directly controlling whether the optimization algorithm converges effectively or fails entirely.
5. **Regularization.** Which regularizer did you use, and did it actually help (with numbers)?
I utilized two forms of regularization: a Dropout layer (p=0.3) within the fully connected classifier block and an Early Stopping monitor. Early Stopping provided a measurable impact by terminating the training loop exactly when validation metrics began degrading (Epoch 8), preventing the validation loss from inflating further and ensuring the model generalized optimally to unseen data.
6. **Cost / benefit.** Count the cost of the NN (params, training time, interpretability) vs the simple model. Is it justified?
The neural network requires 421,642 parameters, demands significantly more computational power and memory to train, and operates entirely as a black box compared to the transparent weight coefficients of a simple linear model. However, because this is an unstructured image dataset where recognizing complex patterns is crucial, the 8% jump in predictive accuracy justifies the elevated computational cost and loss of interpretability.
7. **When DL.** For your task, would you deploy DL or the simpler model? Defend it.
For this specific computer vision task, I would deploy the Deep Learning model over the simpler baseline. Simpler models necessitate flattening the 2D images into 1D vectors, a process that completely destroys inherent spatial relationships such as edges, textures, and geometric shapes. CNNs are explicitly engineered to capture and leverage these spatial hierarchies, making them the definitively correct architecture for image data.
8. **Monday morning.** What would you monitor in production, and what would trigger a retrain?
In a production environment, I would continuously monitor for real-world data drift, such as user-uploaded images that feature different lighting, angles, or noisy backgrounds compared to the pristine Fashion-MNIST dataset. A retrain sequence would be triggered if the model's operational Macro F1 score falls below a predetermined acceptable business threshold (e.g., 88%), or if the business introduces entirely new clothing categories that require expanding the classification output layer.

---

## 4. DL Model Card
# DL Model Card

## 1. Overview
- Option / task / data: Option B / Multi-class Image Classification / Fashion-MNIST dataset.
- Architecture (layers, params): CNN (FashionCNN). 2 Convolutional blocks (Conv2d -> ReLU -> MaxPool2d) + 1 Classifier block (Flatten -> Linear -> ReLU -> Dropout -> Linear). Total Trainable Params: 421,642.

## 2. Setup
- Loss and why (logits handling): CrossEntropyLoss. Handles raw logits directly, combining LogSoftmax and NLLLoss internally for superior numerical stability.
- Optimizer, learning rate, regularizer: Adam optimizer, LR = 1e-3. Regularized using a Dropout layer (p=0.3) and Early Stopping (patience=3).

## 3. Performance
- Simpler-model baseline: ~84% accuracy (e.g., Logistic Regression on flattened images).
- Neural-net test metric: ~92.7% validation accuracy (Best Val Loss: 0.2186).
- Did DL win? By how much, at what cost?: Yes, DL outperformed the baseline by ~8%. The cost is significantly higher computational overhead (420k params vs. 7.8k) and the loss of direct feature interpretability.

## 4. Diagnostics
- Learning curves: where do train and val diverge?: Divergence begins around Epoch 8. Validation loss reaches its minimum and starts to climb, while train loss continues dropping.
- Learning-rate sensitivity: LR 1e-1 is too large (unstable/oscillates), LR 1e-3 is optimal, LR 1e-5 is too small (fails to converge in 10 epochs).

## 5. Decision
- Would you deploy DL or the simpler model here? Defend it.: DL (CNN). Flattening images for simple models destroys critical spatial context. The CNN explicitly maps this spatial data, making the 8% performance boost well worth the extra compute.
- Production: what to monitor, what triggers a retrain.: Monitor real-world data drift (e.g., variable lighting/angles in live user photos). Retrain if macro performance drops below the business threshold or if new clothing classes are introduced.


## 5. Reflection
What surprised you? When, in your mid-term project, would reaching for DL be the right call?
The immediacy with which the training and validation loss curves began to diverge was surprising, clearly illustrating how rapidly deep neural networks can memorize training noise (overfit) if early stopping and regularization constraints are not actively enforced.

When, in your mid-term project, would reaching for DL be the right call?
Reaching for a Deep Learning architecture is the optimal call when the project involves complex, unstructured data (such as images, raw text, or audio) where spatial or sequential hierarchies are fundamentally required to solve the problem, and when maximizing predictive performance takes precedence over strict algorithmic interpretability.

