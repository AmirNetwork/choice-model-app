# config/constants.py
import os

# Folder paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "storage", "uploads")
LOG_FOLDER = os.path.join(BASE_DIR, "storage", "logs")
COMPLETED_MODELS_FOLDER = os.path.join(BASE_DIR, "storage", "completed_models")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"xlsx", "csv"}

# Choice modeling keywords (for quick relevance check)
CHOICE_MODELING_KEYWORDS = [
    "choice model", "logit", "multinomial", "discrete choice",
    "conjoint analysis", "preference", "utility", "willingness to pay", "wtp",
    "market share simulation", "attribute", "level", "respondent", "stated preference",
    "revealed preference", "mixed logit", "latent class", "nested logit",
    "random forest", "gradient boosting", "neural network", "machine learning",
    "feature importance", "predict choice", "classify choice", "model specification"
]

# Specific list of "traditional" model types
TRADITIONAL_MODEL_KEYWORDS = [
    "mixed logit", "multinomial logit", "mnl", "conditional logit",
    "nested logit", "latent class logit", "probit"
]

# Data-Driven Model Sample Code (keys must be lowercase)
DATA_DRIVEN_SAMPLE_CODE = {
    "random forest": """
# Random Forest Classifier Example
# ... (Full code snippet) ...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assuming df, target_variable, feature_columns defined
X = df[feature_columns].copy()
y = df[target_variable].copy()

if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

rf_model = RandomForestClassifier(
    n_estimators=150, random_state=42, class_weight='balanced',
    max_depth=10, min_samples_split=5)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Report:\\n", report)
print("Matrix:\\n", conf_matrix)

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print("Importances:\\n", feature_importance_df.head(10))
""",
    "gradient boosting": """
# Gradient Boosting Classifier Example
# ... (Full code snippet) ...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

X = df[feature_columns].copy()
y = df[target_variable].copy()

if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if not categorical_cols.empty:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    random_state=42, subsample=0.8)

gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Report:\\n", classification_report(y_test, y_pred))

importances = gb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print("Importances:\\n", feature_importance_df.head(10))
""",
    "neural network": """
# Neural Network Example (using Keras/TensorFlow)
# ... (Full code snippet) ...
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X = df[feature_columns].copy()
y = df[target_variable].copy()

if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)
else:
    num_classes = int(y.max() + 1)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_train_processed_check = preprocessor.fit_transform(X_train)
input_dim = X_train_processed_check.shape[1]

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes if num_classes > 2 else 1,
            activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    loss_function = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model

nn_model = build_model(input_dim, num_classes)
nn_model.summary()

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = nn_model.fit(
    X_train_processed, y_train,
    epochs=100, batch_size=32, validation_split=0.2,
    callbacks=[early_stopping], verbose=0
)

loss, accuracy = nn_model.evaluate(X_test_processed, y_test, verbose=0)
y_pred_proba = nn_model.predict(X_test_processed)

if num_classes > 2:
    y_pred = np.argmax(y_pred_proba, axis=-1)
else:
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print(f"Accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")
# You can add classification_report if you import it, e.g. from sklearn.metrics import classification_report
# print(\"Report:\\n\", classification_report(y_test, y_pred))
print(\"(NN Feat Importance is non-trivial to interpret)\")
"""
}

