<h1>📧 Email Spam Classification using PyTorch & NLTK</h1>
    <p>This project classifies emails as <strong>spam</strong> or <strong>not spam</strong> using a deep learning model built with <strong>PyTorch</strong>. It uses <strong>NLTK</strong> for natural language preprocessing and evaluates performance through accuracy metrics and confusion matrix.</p>
    <h2>🔍 Project Overview</h2>
    <ul>
        <li><strong>Goal:</strong> Email text spam detection</li>
        <li><strong>Tech Stack:</strong> PyTorch, NLTK, scikit-learn, matplotlib, seaborn</li>
        <li><strong>Model:</strong> Feedforward ANN</li>
    </ul>
    <h2>🧠 Model Architecture</h2>
    <pre><code>class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
</code></pre>
    <h2>📊 Model Performance</h2>
    <ul>
        <li><strong>Train Accuracy:</strong> 99.91%</li>
        <li><strong>Test Accuracy:</strong> 97.47%</li>
    </ul>
    <h3>📉 Training Loss Curve</h3>
    <img src="https://raw.githubusercontent.com/goldstring/Email-Spam-Detection-Using-NLTK-PYTORCH/refs/heads/main/training_loss.png" alt="Training Loss Curve">
    <h3>📘 Confusion Matrix</h3>
    <img src="https://raw.githubusercontent.com/goldstring/Email-Spam-Detection-Using-NLTK-PYTORCH/refs/heads/main/confustion_matrix.png" alt="Confusion Matrix">
    <h3>📋 Classification Report</h3>
    <table>
        <thead>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Class 0 (Ham)</td>
                <td>0.97</td>
                <td>1.00</td>
                <td>0.98</td>
                <td>619</td>
            </tr>
            <tr>
                <td>Class 1 (Spam)</td>
                <td>0.99</td>
                <td>0.86</td>
                <td>0.92</td>
                <td>131</td>
            </tr>
            <tr>
                <td><strong>Overall Accuracy</strong></td>
                <td colspan="4"><strong>97.47%</strong> (750 samples)</td>
            </tr>
        </tbody>
    </table>
    <p><strong>Test Loss:</strong> 0.4157</p>
    <h2>🔄 Preprocessing</h2>
    <ul>
        <li>Lowercase conversion</li>
        <li>Stopword removal using NLTK</li>
        <li>Punctuation stripping</li>
        <li>TF-IDF vectorization</li>
        <li>80-20 train-test split</li>
    </ul>
    <h2>🧪 How to Run</h2>
    <pre><code># Install dependencies
pip install torch nltk scikit-learn matplotlib seaborn
# Train the model
python train.py
# Evaluate the model
python evaluate.py
</code></pre>
    <h2>📁 Project Structure</h2>
    <pre><code>├── train.py
├── evaluate.py
├── model.py
├── preprocess.py
├── data/
│   └── emails.csv
├── outputs/
│   ├── loss_plot.png
│   └── confusion_matrix.png
├── README.html
</code></pre>
    <h2>🚀 Future Improvements</h2>
    <ul>
        <li>Improve spam recall by balancing data</li>
        <li>Upgrade to LSTM or transformer models</li>
        <li>Integrate with email clients for real-time spam detection</li>
    </ul>
