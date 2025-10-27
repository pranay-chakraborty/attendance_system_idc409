import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.prepare_data import prepare_data

MODEL_PATH = 'models/eigenface_model.yml'

def generate_multiclass_roc():
    """
    Generates One-vs-Rest (OvR) ROC curves for the multi-class face recognizer.
    """
    print("--- Generating Multi-Class One-vs-Rest ROC Curves ---")

    _, test_faces, _, y_true = prepare_data()
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)
    
    classes = np.unique(y_true)
    
    y_true_binarized = label_binarize(y_true, classes=classes)
    n_classes = y_true_binarized.shape[1]

    y_scores = []
    for test_image in test_faces:
        _, confidence = face_recognizer.predict(test_image)
        y_scores.append(-confidence)

    y_pred_labels = [face_recognizer.predict(img)[0] for img in test_faces]
    y_pred_binarized = label_binarize(y_pred_labels, classes=classes)
    y_scores_adjusted = y_pred_binarized * np.array(y_scores)[:, np.newaxis]
    y_scores_adjusted[y_scores_adjusted == 0] = -99999
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores_adjusted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class One-vs-Rest ROC Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True) 
    plt.show()


if __name__ == '__main__':
    generate_multiclass_roc()