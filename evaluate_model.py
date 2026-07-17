import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score

def evaluate():
    model = joblib.load('rf_model.pkl')
    X_test, y_test = joblib.load('test_data.pkl')
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc * 100:.2f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm = confusion_matrix(y_test, preds)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    cmd.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix')
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    score = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='orange', label=f'AUC = {score:.2f}')
    axes[1].plot([0, 1], [0, 1], color='blue', linestyle='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()