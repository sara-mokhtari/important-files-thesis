#!/usr/bin/env python
'''
Main functionalities:
- Loads and preprocesses SAXS data from CSV and NPZ files, including duplicate removal, logarithmic transformation, and derivative computation.
- Encodes particle shape labels and prepares PyTorch tensors for model input.
- Applies data augmentation techniques to improve model robustness : apply masks to simulate experimental data intervals, with a controlled fraction of full-length profiles.
- Defines a multi-task neural network architecture (SAXSMultiTask) with a feature extractor, classifier, and regressor for simultaneous shape classification and size regression.
- Trains the model using stratified K-fold cross-validation, with early stopping and learning rate scheduling.
- Evaluates model performance, plots training history and regression predictions, and computes average confusion matrices.
- Saves the trained model and label encoder for future inference.
Key functions and classes:
- supprimer_doublons_sans_casser_forme: Removes duplicates from a DataFrame, even with list or array columns.
- prepare_saxs_data: Prepares SAXS data and labels for model training.
- SAXSFeatureExtractor: Neural network module for extracting features from SAXS data.
- SAXSMultiTask: Multi-task model for classification and regression.
- train_saxs_model: Trains and validates the model using cross-validation.
- plot_training_history: Plots training and validation loss and accuracy.
- plot_predictions_size: Plots predicted vs. true size values.
- save_complete_pipeline: Saves the trained model and label encoder.
- main_pipeline: Orchestrates the full data preparation, training, evaluation, and saving pipeline.
Requirements:
- PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib
Usage:
- Adjust file paths for CSV and NPZ data as needed.
- Run the script to train and evaluate the model on the provided dataset.
'''
# coding: utf-8

# # Test 1 avec GPU

# In[5]:


from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas as pd
import torch
import ast
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
#torch.cuda.empty_cache()

if __name__ == "__main__":
    print('Script starting !', flush = True)
    # Device (GPU obligatoire)
    if not torch.cuda.is_available():
        raise SystemError("GPU CUDA unvailable, programm stopped.")
    device = torch.device('cuda')
    print(f"Using device: {device}", flush=True)
    print("Device used:", torch.cuda.current_device(), torch.cuda.get_device_name(0))


    # Charger le fichier .csv et le mettre en forme
    df = pd.read_csv("csv_final/df_completed_22446_lines_2025-09-26.csv")  # Remplace par ton chemin avec :  "df_final/df_completed_XXXXX_lines_YYYY-MM-DD.csv"
    data = np.load('csv_final/iq_all_2025-09-26.npz', allow_pickle = True) 

def supprimer_doublons_sans_casser_forme(df):
    """
    Supprime les doublons dans un DataFrame même si certaines colonnes contiennent
    des listes ou numpy arrays, en comparant la représentation en chaînes.
    La forme du DataFrame n'est pas modifiée.
    """
    # Créer une série où chaque ligne est convertie en string
    lignes_str = df.applymap(lambda x: str(x)).astype(str).agg('_'.join, axis=1)
    # Trouver les indices des lignes uniques selon cette représentation
    indices_uniques = lignes_str.drop_duplicates().index
    # Sélectionner uniquement ces lignes dans le DataFrame d'origine
    return df.loc[indices_uniques].reset_index(drop=True)

if __name__ == "__main__": 
    df = supprimer_doublons_sans_casser_forme(df)

def prepare_saxs_data(df,data, apply_mask = True):
    """
    Prépare les données en respectant la structure SAXS/WAXS
    """
    
    # Extraction of the data
    
    # 1. Shapes and sizes 
    
    # Shapes
    shapes = df["shape"].tolist()
    # plot shape distribution
    plt.figure(figsize=(10,5))
    plt.hist(shapes, bins=len(set(shapes)))
    plt.xlabel("Shape")
    plt.show()
    plt.savefig('hist_shape.png')
    plt.close()
    
    # Sizes
    plt.figure(figsize=(10,5))
    size = np.array(df["circumradius_A"])
    print("Size of the size array is: ", len(size))
    # plot size distribution
    plt.hist(df["circumradius_A"], bins=50)
    plt.xlabel("Taille (Å)")
    plt.show()
    plt.savefig('hist_size.png')
    plt.close()

    # Encodage des labels
    # Sizes
    scaler_size = StandardScaler()
    size = scaler_size.fit_transform(size.reshape(-1, 1)).flatten()
    # Shapes
    scaler_shape = LabelEncoder()
    labels = scaler_shape.fit_transform(shapes)
    print('labels',labels)

 
    # 2. Retreive Iq data
    print('Starting to retreive Iq data')
    print('Keys of the npz found: ', data.files)  # ex: ['iq_saxs', 'iq_waxs','q']

    saxs_array = [np.array(arr) for arr in data['iq_saxs']]  # list of arrays (each shape ≈ 1599)
    q_array    = [np.array(arr) for arr in data['q']]        # list of arrays (each shape ≈ 20000)
    saxs_array = np.stack(saxs_array)   # (n_files, 1599)
    q_array    = np.stack(q_array)      # (n_files, 20000)

    print("Shape de saxs_array (should be nfiles, 1599): ",saxs_array.shape) # (nfiles, 1599)
    print("Shape de q_array (should be nfiles, 20000): ", q_array.shape)

    # 3. Transform Iq data : separate q data, interpolation in WAXS (divided by 10), logarithm, derivative
    print('Starting to transform Iq data')
    # Separate q in q_saxs and q_waxs
    index=[index for index,value in enumerate(q_array[0]) if np.isclose(value, 1.6)]
    q_saxs=q_array[0][:index[0]]
    q_saxs=q_saxs.reshape(1599)
    print("Shape of q_saxs is: ", q_saxs.shape)
    # Compute logarithm
    saxs_array = np.log10(saxs_array)   # (n_files, 1599)
    # q_array    = np.log10(q_array)      # (n_files, 20000)     


     # std each profile
    mean = np.mean(saxs_array, axis=1, keepdims=True)
    std = np.std(saxs_array, axis=1, keepdims=True)
    saxs_array = (saxs_array - mean) / std
    print("SAXS array after log and standardization shape:", saxs_array.shape)
    
    # Compute derivate
    def compute_derivative(signal,q):
        if signal.shape[-1] == 1: 
            signal = signal.squeeze(-1)  # from (nfiles, 1499,1) to  (nfiles, 1499)
        # Calcul de la dérivée sur chaque ligne (échantillon)
        derivs = np.array([np.gradient(y, q) for y in signal])  # shape (nfiles, 1499)
        # Remettre une 3e dimension pour le CNN
        X_deriv = derivs.reshape(derivs.shape[0], derivs.shape[1], 1)# from (nfiles, 1499) to  (nfiles, 1499,1)
        return X_deriv    
    
    def standardize_derivative(X_deriv):
        mean = np.mean(X_deriv, axis=1, keepdims=True)
        std = np.std(X_deriv, axis=1, keepdims=True)
        return (X_deriv - mean) / std

    X_saxs_deriv = compute_derivative(saxs_array,q_saxs)
    print('X_saxs_deriv shape:',X_saxs_deriv.shape)
    X_saxs_deriv_std = standardize_derivative(X_saxs_deriv)
    print('X_saxs_deriv_std shape',X_saxs_deriv_std.shape)
    saxs_array = saxs_array[..., np.newaxis]
    print('saxs_array shape that must be equal to the deriv_std:',saxs_array.shape)
    saxs_array = np.concatenate([saxs_array, X_saxs_deriv_std], axis=-1)

    print("Fin de la transformation des entrées Iq, d(Iq)")


    # Creation of the PyTorch tensor 
    X = saxs_array
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (n, channels=2, L)
    X_tensor_masked = torch.zeros((X_tensor.shape[0], 3, X_tensor.shape[2]), dtype=X_tensor.dtype)  # Placeholder for masked data with 3 channels
    y = torch.LongTensor(labels).to(device)

    # Add mask to simulate experimental data intervals
    if apply_mask:

        # Fraction of the full length profiles
        full_length_fraction = 0.2 # 20 % of the profiles will have the full length
        n_total = X_tensor.shape[0]
        n_full = int(n_total * full_length_fraction)
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        full_indices = indices[:n_full]
        masked_indices = indices[n_full:]
        n_samples = len(masked_indices)
        print(f"Total samples: {n_total}, Full length samples: {n_full}, Masked samples: {n_samples}")

        # Put 1 in the mask for the full length profiles
        for i in full_indices:
            X_tensor_masked[i, 0, :] = X_tensor[i, 0, :]
            X_tensor_masked[i, 1, :] = X_tensor[i, 1, :]
            X_tensor_masked[i, 2, :] = 1.0  # Canal de masque rempli de 1
        
        

        for idx, i in enumerate(masked_indices):
            true_size = df["circumradius_A"].iloc[i]
            if true_size < 10 :
                w = [[0.001, 1.6], [0.1, 1.6]]

            elif true_size >= 10 and true_size < 15  :
                w = [[0.001, 0.8], [0.001, 1.6], [0.01, 1]]

            elif true_size >= 15 and true_size < 30 :
                w = [[0.003, 0.5], [0.01, 1], [0.001, 1.6]]

            else : # true_size >= 30
                w = [[0.001, 0.3], [0.001, 1.6]]

            w = w[np.random.randint(len(w))]  # Randomly choose one window among the possible ones
            mask = ((q_saxs >= w[0]) & (q_saxs <= w[1]))
            X_tensor_masked[i, 0, :] = X_tensor[i, 0, :] * torch.tensor(mask, dtype=X_tensor.dtype)
            X_tensor_masked[i, 1, :] = X_tensor[i, 1, :] * torch.tensor(mask, dtype=X_tensor.dtype)
            X_tensor_masked[i, 2, :] = torch.tensor(mask, dtype=X_tensor.dtype)
            
        print("Mask applied with controlled fixed windows and full length profiles.")

   
    # if masked not applied, keep the same shape but add a channel of 1 
    else:
        X_tensor_masked = torch.zeros((X_tensor.shape[0], 3, X_tensor.shape[2]), dtype=X_tensor.dtype)
        X_tensor_masked[:, 0, :] = X_tensor[:, 0, :]
        X_tensor_masked[:, 1, :] = X_tensor[:, 1, :]
        X_tensor_masked[:, 2, :] = 1.0  # Canal de masque rempli de 1
        print("No mask applied, added channel of ones.")

    return X_tensor_masked, y, scaler_shape, scaler_size, size

class SAXSFeatureExtractor(nn.Module):
    def __init__(self, saxs_length=1600,  n_features=256):
        super().__init__()

        # Branche SAXS uniquement
        self.saxs_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )

        self.saxs_feature_mlp = nn.Sequential(
            nn.Linear(256 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_features),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        saxs_data = x[:, :, :1600]  # [batch, 2, 1600]
        saxs_features = self.saxs_branch(saxs_data)
        saxs_flat = saxs_features.view(batch_size, -1)
        saxs_feats = self.saxs_feature_mlp(saxs_flat)
        final_features = self.combined_mlp(saxs_feats)
        return final_features

class SAXSMultiTask(nn.Module):
    """
    Modèle complet avec extracteur de features + régression

    Analyse ta courbe (tes données SAXS+WAXS) pour en extraire un vecteur de caractéristiques (features).
    Ces caractéristiques sont ensuite utilisées pour effectuer une régression afin de prédire des valeurs continues.
    """
    def __init__(self, saxs_length=1500, n_features=256, output_dim=1, n_classes=13):
        super().__init__()

        self.feature_extractor = SAXSFeatureExtractor(
            saxs_length, n_features
        )        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)  # Output dimension is 1 for single value regression
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        reg_output = self.regressor(features)
        return class_logits, reg_output
    

def train_saxs_model(X, y, size, scaler_shape, scaler_size, epochs=200, patience=50):#400 et 40
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)

    n_classes = len(np.unique(y_np))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    conf_matrix_total = np.zeros((n_classes, n_classes))

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y_np)):
        print(f"\n=== Fold {fold + 1} ===")

        # Split train+val et test
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        size_train_val, size_test = size[train_val_idx], size[test_idx]

        # Split interne train/val (80/20 par exemple)
        val_size = int(0.2 * len(X_train_val))
        X_val, y_val = X_train_val[:val_size], y_train_val[:val_size]
        X_train, y_train = X_train_val[val_size:], y_train_val[val_size:]
        size_val, size_train = size_train_val[:val_size], size_train_val[val_size:]

        # To device
        X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
        y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)
        size_train, size_val, size_test = (
            torch.tensor(size_train, dtype=torch.float32).to(device),
            torch.tensor(size_val, dtype=torch.float32).to(device),
            torch.tensor(size_test, dtype=torch.float32).to(device),
        )

        # Modèle
        model = SAXSMultiTask(
            saxs_length=1500,
            n_features=256,
            n_classes=n_classes,
            output_dim=1
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
        criterion_reg = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()

        batch_size = 32
        train_loader = DataLoader(TensorDataset(X_train, size_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, size_val, y_val), batch_size=batch_size, shuffle=False)

        train_losses, val_losses, val_accuracies = [], [], []
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # === Entraînement ===
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            total = 0
            correct = 0

            for xb, size_b, yb in train_loader:
                xb, size_b, yb = xb.to(device), size_b.to(device), yb.to(device)
                optimizer.zero_grad()

                class_logits, reg_output = model(xb)
                loss_reg = criterion_reg(reg_output.squeeze(), size_b)
                loss_class = criterion_class(class_logits, yb)
                loss = loss_reg + loss_class

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * xb.size(0)
                total += yb.size(0)
                # Accuracy classification
                preds = class_logits.argmax(dim=1)
                correct += (preds == yb).sum().item()

            train_loss /= total
            train_acc = correct / total if total > 0 else 0

            # === Validation ===
            model.eval()
            val_loss = 0.0
            total_val = 0
            val_correct = 0
            with torch.no_grad():
                for xb, size_b, yb in val_loader:
                    xb, size_b, yb = xb.to(device), size_b.to(device), yb.to(device)
                    class_logits, reg_output = model(xb)
                    loss_reg = criterion_reg(reg_output.squeeze(), size_b)
                    loss_class = criterion_class(class_logits, yb)
                    loss = 1 * loss_reg + 10 * loss_class  # change the weight of each loss if needed
                    val_loss += loss.item() * xb.size(0)
                    total_val += yb.size(0)
                    preds = class_logits.argmax(dim=1)
                    val_correct += (preds == yb).sum().item()
            val_loss /= total_val
            val_acc = val_correct / total_val if total_val > 0 else 0

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')
            else:
                epochs_without_improvement += 1

            if epoch % 25 == 0 or epochs_without_improvement == 0:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping après {epoch} epochs")
                break

        # === Évaluation test ===
        model.load_state_dict(torch.load(f'best_model_fold{fold+1}.pth'))
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            size_test = size_test.to(device)
            y_test = y_test.to(device)
            class_logits, reg_output = model(X_test)
            test_class_preds = class_logits.argmax(dim=1).cpu().numpy()
            test_reg_preds = reg_output.squeeze().cpu().numpy()
            test_true_classes = y_test.cpu().numpy()
            test_true_sizes = size_test.cpu().numpy()

        # Plot training metrics
        plot_training_history(train_losses, val_losses, val_accuracies, fold)
        plot_predictions_size(test_true_sizes, test_reg_preds, scaler_size, title=f'Fold {fold + 1} Regression: Predictions vs True', filename=f'predictions_vs_true_fold{fold + 1}.png')
        print(f"Fold {fold + 1} evaluation complete.")
        conf_matrix = confusion_matrix(test_true_classes, test_class_preds)
        conf_matrix_total += conf_matrix

    # Matrice de confusion moyenne
    conf_matrix_avg = conf_matrix_total / n_splits
    print(f"Matrice de confusion moyenne sur {n_splits} folds :")
    print(conf_matrix_avg)
    # Sauvegarde pdf
    classes = scaler_shape.classes_
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_avg, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Matrice de confusion moyenne sur {n_splits} folds")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{n_splits}kfold_avg_deriv2_2025_09_26.pdf")
    plt.close()

    return model, test_class_preds, test_true_classes, test_reg_preds, test_true_sizes  # dernier modèle entraîné

def plot_training_history(train_losses, val_losses, val_accuracies,fold):
    """
    Affiche l'historique d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Pertes
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Évolution des pertes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Précision
    ax2.plot(val_accuracies, label='Val Accuracy', alpha=0.8, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Précision de validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Métriques_tristan_deriv2_2025_09_26.pdf')
    plt.show()


def plot_predictions_size(y_true, y_pred, scaler_size, title='Prédictions vs Vérités', filename='predictions_vs_true.png'):
    """
    Affiche les prédictions vs les valeurs réelles
    """
    y_pred = scaler_size.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true = scaler_size.inverse_transform(y_true.reshape(-1, 1)).flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # ligne y=x
    plt.xlabel('Real sizes')
    plt.ylabel('Predicted sizes')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.show()



def save_complete_pipeline(cnn_model, scaler_shape, scaler_size):
    """
    Sauvegarde complète du pipeline
    """
  
    # Sauvegarder le modèle CNN
    params = {
        'saxs_length': 1600,
        'q_saxs_min': 0.001,
        'q_saxs_max': 1.6,
        'n_features': 256,
        'output_dim': 1,
        'n_classes': len(scaler_shape.classes_)
    }
    torch.save(params, 'cnn_model_params.pth')
    torch.save(cnn_model, 'cnn_reg_classif_saxs_complete_model.pth')  # modèle complet
    # Sauvegarder les encodeurs
    joblib.dump(scaler_shape, 'scaler_shape_deriv.pkl')
    joblib.dump(scaler_size, 'scaler_size_deriv.pkl')
    print("Pipeline complet sauvegardé!")


# ===== EXEMPLE D'UTILISATION COMPLÈTE =====
def main_pipeline(df):
    """
    Pipeline complet d'entraînement et d'évaluation
    """
    print("=== PRÉPARATION DES DONNÉES ===")
    X, y, scaler_shape, scaler_size, size = prepare_saxs_data(df,data)
    print(f"Données préparées: {X.shape}", flush = True)
    print(f"Classes: {scaler_shape.classes_}", flush = True)
   
    with torch.cuda.device(0): 

        print("\n=== ENTRAÎNEMENT DU CNN ET EVALUATION ===")
        cnn_model, y_class_pred_cnn, y_class_true_cnn, y_reg_pred_cnn, y_reg_true_cnn = train_saxs_model(X, y, size, scaler_shape, scaler_size, epochs=200) 


    print("\n=== SAUVEGARDE ===")
    save_complete_pipeline(cnn_model, scaler_shape, scaler_size)

    return cnn_model, scaler_shape, scaler_size

# Entraînement complet
if __name__ == "__main__":
    cnn_model,  scaler_shape, scaler_size = main_pipeline(df)




