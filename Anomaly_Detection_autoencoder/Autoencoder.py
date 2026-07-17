import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return input_error

class Activation:
    def __init__(self, activation_fn, activation_prime_fn):
        self.activation = activation_fn
        self.activation_prime = activation_prime_fn

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def sigmoid(x):

    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

abnormal_pump_path = "data/pump/id_00/abnormal"
normal_pump_path = "data/pump/id_00/normal"

def load_audio_files(path, label):
    audio_files = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            file_path = os.path.join(path, filename)
            audio, sample_rate = librosa.load(file_path, sr=None)
            audio_files.append(audio)
            labels.append(label)
    return audio_files, labels, sample_rate

print("Loading Audio Files...")
abnormal_audio, abnormal_labels, _ = load_audio_files(abnormal_pump_path, label=1)
normal_audio, normal_labels, sample_rate = load_audio_files(normal_pump_path, label=0)

def extract_features(audio_data, sample_rate):
    features = []
    for audio in audio_data:

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
        spec_roll = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0])
        spec_cont = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0])

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        autocorr = np.mean(librosa.autocorrelate(audio))
        
        all_feats = np.concatenate([mfccs_processed, [spec_cent, spec_roll, spec_cont, zcr, autocorr]])
        features.append(all_feats)
    return np.array(features)

print("Extracting Features...")
normal_features = extract_features(normal_audio, sample_rate)
abnormal_features = extract_features(abnormal_audio, sample_rate)

X_train, X_val = train_test_split(normal_features, test_size=0.2, random_state=42)
X_test = abnormal_features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_combined_test = np.concatenate((X_val_scaled, X_test_scaled))
y_combined_test = np.concatenate((np.zeros(len(X_val_scaled)), np.ones(len(X_test_scaled))))


def train_autoencoder(X_train, X_val, epochs=150, batch_size=64, learning_rate=0.01):
    input_dim = X_train.shape[1]

    network = [
        Dense(input_dim, 64),
        Activation(relu, relu_prime),
        Dense(64, 32),
        Activation(relu, relu_prime),
        Dense(32, 64),
        Activation(relu, relu_prime),
        Dense(64, input_dim),
        Activation(sigmoid, sigmoid_prime)
    ]
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        
        for i in range(0, X_train.shape[0], batch_size):
            batch = X_train_shuffled[i:i+batch_size]

            output = batch
            for layer in network:
                output = layer.forward(output)

            error = mse_prime(batch, output)
            for layer in reversed(network):
                error = layer.backward(error, learning_rate)

        if (epoch + 1) % 10 == 0:
            val_out = X_val
            for layer in network:
                val_out = layer.forward(val_out)
            val_loss = mse(X_val, val_out)
            print(f"Epoch {epoch+1}/{epochs}, Validation MSE: {val_loss:.6f}")
            
    return network


def evaluate_network(network, X_combined_test):
    print("Evaluating...")

    reconstructed_combined = X_combined_test
    for layer in network:
        reconstructed_combined = layer.forward(reconstructed_combined)
        
    mse_combined = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=1)
    
    precisions, recalls, thresholds = precision_recall_curve(y_combined_test, mse_combined)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_predictions = (mse_combined > optimal_threshold).astype(int)
    
    optimal_accuracy = accuracy_score(y_combined_test, optimal_predictions)
    optimal_precision = precision_score(y_combined_test, optimal_predictions)
    optimal_recall = recall_score(y_combined_test, optimal_predictions)
    optimal_f1 = f1_score(y_combined_test, optimal_predictions)
    
    cm = confusion_matrix(y_combined_test, optimal_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- RESULTS ---")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"Accuracy:  {optimal_accuracy:.4f}")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Recall:    {optimal_recall:.4f}")
    print(f"F1 Score:  {optimal_f1:.4f}")
    
    print(f"\n--- CONFUSION MATRIX ---")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives:  {tn}")
    print(f"False Negatives: {fn}")

if __name__ == "__main__":
    trained_net = train_autoencoder(X_train_scaled, X_val_scaled)
    evaluate_network(trained_net, X_combined_test)