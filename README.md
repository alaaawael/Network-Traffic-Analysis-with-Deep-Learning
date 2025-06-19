# Network-Traffic-Analysis-with-Deep-Learning
This project implements a network traffic analysis system using deep learning to classify network packets as normal or malicious. It leverages the KDD Cup dataset for training and uses Scapy for real-time packet sniffing.
Features

Loads and preprocesses the KDD Cup 10% dataset.
Builds and trains a neural network to classify network traffic.
Generates adversarial examples using the Fast Gradient Method (FGM).
Performs real-time packet capture and prediction using Scapy.
Logs captured packets with predictions to a file.

Requirements

Python 3.x
TensorFlow
Scapy
Pandas
NumPy
Matplotlib
Scikit-learn
Ensure all dependencies are installed via pip install -r requirements.txt.

Installation

Clone the repository:git clone https://github.com/alaaawael/network-traffic-analysis.git
cd network-traffic-analysis


Install the required packages:pip install -r requirements.txt


Place the KDD Cup dataset (kddcup.data_10_percent_corrected) in the specified path or update the path in the code.

Usage

Run the script:python SourceCode.py


The script will load the dataset, train the model, and start sniffing packets on the specified interface (e.g., "Wi-Fi") for 30 seconds.
Packet predictions and logs will be printed to the console and saved to packet_log.txt.

Dataset

The project uses the KDD Cup 1999 dataset (10% subset).
Download it from KDD Cup 1999 and place it in the specified directory.

Output

Training and validation accuracy/loss plots are displayed.
Real-time packet predictions are logged to packet_log.txt.

Contributing
Feel free to fork this repository and submit pull requests. Please ensure any changes are tested and documented.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

KDD Cup 1999 dataset contributors.
Scapy and TensorFlow communities for their excellent tools.

