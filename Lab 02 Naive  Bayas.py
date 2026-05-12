from sklearn.naive_bayes import GaussianNB

# Implementation of Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluation
print("--- Naive Bayes Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix Visualization
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6,4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Authentic', 'Forged'], 
            yticklabels=['Authentic', 'Forged'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Naive Bayes')
plt.show()
