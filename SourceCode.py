import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP, get_if_list
import datetime
import warnings
import os

# Suppress MinMaxScaler warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Ensure TensorFlow is in eager mode
if not tf.executing_eagerly():
    print("Eager execution was disabled. Enabling it now...")
    tf.config.run_functions_eagerly(True)
else:
    print("Eager execution is already enabled.")

# Verify TensorFlow version
print("TensorFlow version:", tf.__version__)

# Define column names for KDD Cup dataset
cols = """duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""
columns = [c.strip() for c in cols.split(',\n') if c.strip()]
columns.append('target')
columns.append('extra')

# Load dataset (Requirement 1)
try:
    path = r"C:\Users\Alaa\Downloads\kddcup.data_10_percent_corrected"
    df = pd.read_csv(path, names=columns, on_bad_lines='skip')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Clean the target column
df['target'] = df['target'].astype(str).str.strip('.')

# Define expected numeric columns
numeric_cols = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Convert numeric columns, coercing invalid values to NaN, then fill with 0
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Preprocess dataset (Requirement 1)
attacks_types = {
    'normal': 'normal', 'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l', 'guess_passwd': 'r2l',
    'imap': 'r2l', 'ipsweep': 'probe', 'land': 'dos', 'loadmodule': 'u2r', 'multihop': 'r2l', 'neptune': 'dos',
    'nmap': 'probe', 'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos', 'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe',
    'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos', 'warezclient': 'r2l', 'warezmaster': 'r2l'
}
df['Attack Type'] = df['target'].map(attacks_types)

# Ensure Attack Type is uniformly strings
df['Attack Type'] = df['Attack Type'].astype(str)

#  Check for mixed types in Attack Type
if df['Attack Type'].apply(lambda x: isinstance(x, (int, float))).any():
    print("Warning: Mixed types detected in 'Attack Type' column. Converting all to strings.")
    df['Attack Type'] = df['Attack Type'].astype(str)

# Drop unnecessary columns
df.drop(['num_root', 'srv_serror_rate', 'srv_rerror_rate', 'dst_host_srv_serror_rate',
         'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
         'dst_host_same_srv_rate', 'service', 'extra', 'target'], axis=1, inplace=True)

# Encode categorical features
df['protocol_type'] = df['protocol_type'].map({'icmp': 0, 'tcp': 1, 'udp': 2})
df['flag'] = df['flag'].map({
    'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4,
    'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10
})

# Handle missing values
df['protocol_type'] = df['protocol_type'].fillna(0)
df['flag'] = df['flag'].fillna(0)
df = df.fillna(0)

# Prepare features and labels
x = df.drop(['Attack Type'], axis=1)
feature_names = x.columns.tolist()
x = x.values

# Normalize features
sc = MinMaxScaler()
x_scaled = sc.fit_transform(x)

# Encode labels
y = df['Attack Type']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_categorical, test_size=0.33, random_state=42)

# Build deep learning model (Requirement 2)
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (Requirement 2)
try:
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)
    print("Training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Evaluate model (Requirement 2)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history (Requirement 2)
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Implement a simple Fast Gradient Method (FGM) attack manually (Requirement 5)
def fgm_attack(model, x, y, eps=0.1):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)
    
    gradient = tape.gradient(loss, x_tensor)
    signed_grad = tf.sign(gradient)
    x_adv = x_tensor + eps * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()

# Generate adversarial examples
try:
    x_test_adv = fgm_attack(model, x_test, y_test, eps=0.1)
    print("Adversarial examples generated successfully.")
except Exception as e:
    print(f"Error during FGM attack: {e}")
    raise

# Evaluate model on adversarial examples
adv_loss, adv_accuracy = model.evaluate(x_test_adv, y_test, verbose=0)
print("Adversarial Test Loss:", adv_loss)
print("Adversarial Test Accuracy:", adv_accuracy)

# Real-time packet analysis with Scapy (Requirement 3)
def preprocess_packet(pkt):
    try:
        proto = 1 if pkt.haslayer(TCP) else (2 if pkt.haslayer(UDP) else 0)
        length = len(pkt)
        src_bytes = len(pkt[TCP].payload) if pkt.haslayer(TCP) else (len(pkt[UDP].payload) if pkt.haslayer(UDP) else 0)
        dst_bytes = 0
        land = 1 if IP in pkt and pkt[IP].src == pkt[IP].dst else 0

        # Initialize feature vector with feature names
        features_dict = {name: 0 for name in feature_names}
        features_dict['protocol_type'] = proto
        features_dict['flag'] = 0
        features_dict['src_bytes'] = src_bytes
        features_dict['dst_bytes'] = dst_bytes
        features_dict['land'] = land
        features_df = pd.DataFrame([features_dict])

        # Scale features using the fitted scaler
        features_scaled = sc.transform(features_df)
        prediction = model.predict(features_scaled, verbose=0)
        predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]
        print(f"Packet captured - Predicted: {predicted_label}")
        log_packet(pkt, predicted_label)  # Log the packet (Requirement 4)
    except Exception as e:
        print("Error processing packet:", e)

# Logging functionality (Requirement 4) 
def log_packet(pkt, label):
    log_file = r"C:\Users\Alaa\Downloads\packet_log.txt" 
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now()} | {pkt.summary()} | Predicted: {label}\n")
            f.flush()  # Force flush to ensure data is written immediately
        print(f"Logged packet to {log_file}")
    except Exception as e:
        print(f"Error writing to log file: {e}")

# Start sniffing (Requirement 3)
def start_sniffing():
    try:
        # List available network interfaces
        print("Available network interfaces:", get_if_list())
        
        interface = "Wi-Fi" 
        print(f"Starting live packet capture on interface '{interface}' (timeout 30 seconds)...")
        # Sniff with a timeout to avoid indefinite waiting
        sniff(iface=interface, prn=preprocess_packet, store=0, timeout=30)
        print("Sniffing stopped after timeout.")
    except Exception as e:
        print(f"Error during sniffing: {e}")
        raise

# Start sniffing after training
start_sniffing()