import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import coupled sensors from damage mappings
from src.data_processing.damage_mappings import COUPLED_SENSORS

class GearboxCNNLSTM(nn.Module):
    def __init__(self, window_size=256, lstm_hidden_size=32, num_lstm_layers=1, num_sensors=8, dropout_rate=0.3):
        super().__init__()
        self.num_samples = 0
        self.num_sensors = num_sensors
        self.dropout_rate = dropout_rate
        self.mc_dropout = False  # Flag for Monte Carlo dropout during inference
        self.lstm_hidden_size = lstm_hidden_size
        
        # Physical frequency bands for gear mesh frequencies and their harmonics
        # Used to enhance feature extraction with domain knowledge
        self.gear_mesh_frequencies = [152.3, 789.5, 2340.7]  # Hz
        
        # Sensor-specific feature extraction
        self.sensor_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2),  # Add dropout to CNNs
                nn.MaxPool1d(4)
            ) for _ in range(num_sensors)
        ])
        
        # Enhanced frequency-domain feature extraction
        self.freq_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 4, kernel_size=7, padding=3),  # Wider kernel for better frequency resolution
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=4),  # Depthwise conv for feature refinement
                nn.BatchNorm1d(4),
                nn.ReLU(),
                nn.MaxPool1d(4)
            ) for _ in range(num_sensors)
        ])
        
        # Adaptive pooling for each sensor branch
        self.adaptive_pool = nn.AdaptiveMaxPool1d(16)
        
        # Sensor attention mechanism
        self.sensor_attention = nn.Sequential(
            nn.Linear(8 * 16, 32),
            nn.Dropout(dropout_rate),  # Add dropout to attention
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # LSTM for temporal analysis
        self.lstm_input_features = 8 * num_sensors # 8 output channels from each sensor_cnn branch
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features, 
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            batch_first=True # LSTM expects (batch, seq_len, features)
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 16),
            nn.Dropout(dropout_rate),  # Add dropout to attention
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
        # Fault detection head
        self.fault_detector = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Increased dropout
            nn.Linear(32, 1)
        )
        
        # Individual sensor feature extractors
        self.sensor_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8 * 16, 32),
                nn.LayerNorm(32),  # Added layer norm
                nn.ReLU(),
                nn.Dropout(dropout_rate/2)
            ) for _ in range(num_sensors)
        ])
        
        # Independent sensor anomaly detection heads
        self.sensor_anomaly_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_hidden_size*2 + 32, 64),  # Reduced input size (removed global sensor info)
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2),
                nn.Linear(32, 1)
            ) for _ in range(num_sensors)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout for inference uncertainty estimation"""
        self.mc_dropout = True
    
    def disable_mc_dropout(self):
        """Disable Monte Carlo dropout (normal inference mode)"""
        self.mc_dropout = False
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Set all dropout layers to training mode for MC dropout if enabled
        if self.mc_dropout:
            def apply_dropout_training(m):
                if isinstance(m, nn.Dropout):
                    m.train()
            self.apply(apply_dropout_training)
        
        # Sensor data is the entire input x, as RPM is removed
        sensor_data = x  # (batch_size, time_steps, num_sensors=8)
        
        sensor_cnn_outputs = [] # Store outputs from sensor_cnns after adaptive pooling
        raw_sensor_features_for_anomaly_heads = [] # Store features for anomaly heads (before attention)
        
        for i in range(self.num_sensors):
            # Extract single sensor data and permute: (batch_size, 1, time_steps)
            single_sensor = sensor_data[:, :, i:i+1].permute(0, 2, 1) 
            
            # Apply sensor-specific CNN for time-domain features
            # Output: (batch_size, 8 cnn_output_channels, pooled_length)
            sensor_cnn_feat = self.sensor_cnns[i](single_sensor)
            sensor_cnn_feat_pooled = self.adaptive_pool(sensor_cnn_feat) # (batch_size, 8, adaptive_pool_cnn_out_size)
            sensor_cnn_outputs.append(sensor_cnn_feat_pooled)
            
            # Store raw pooled features for later use in anomaly detection heads
            # These are flattened: (batch_size, 8 * adaptive_pool_cnn_out_size)
            raw_sensor_features_for_anomaly_heads.append(sensor_cnn_feat_pooled.view(batch_size, -1))
            
            # Frequency-domain feature extraction (placeholder for now - not directly combined into LSTM for simplicity)
            # freq_feat = self.freq_extractors[i](single_sensor)
            # freq_feat_pooled = self.adaptive_pool(freq_feat) # (batch_size, 4, adaptive_pool_cnn_out_size)
            # freq_features_list.append(freq_feat_pooled.view(batch_size, -1))

        # Sensor Attention (applied to the raw_sensor_features_for_anomaly_heads)
        sensor_attention_logits = []
        for i in range(self.num_sensors):
            attention_logit = self.sensor_attention(raw_sensor_features_for_anomaly_heads[i])
            sensor_attention_logits.append(attention_logit)
        
        sensor_attention_weights = F.softmax(torch.cat(sensor_attention_logits, dim=1), dim=1) # (batch_size, num_sensors)
        
        # Weighted combination of sensor_cnn_outputs for LSTM
        # Each sensor_cnn_output is (batch, 8, adaptive_pool_cnn_out_size)
        # We want to weight these and concatenate along channel dim (dim=1)
        weighted_features_for_lstm = []
        for i in range(self.num_sensors):
            # Expand attention_weights to match feature dimensions for broadcasting
            # attention_weights[:, i:i+1] is (batch_size, 1)
            # .unsqueeze(-1) makes it (batch_size, 1, 1) to multiply with (batch_size, 8, adaptive_pool_cnn_out_size)
            weighted_feat = sensor_cnn_outputs[i] * sensor_attention_weights[:, i:i+1].unsqueeze(-1)
            weighted_features_for_lstm.append(weighted_feat)
        
        # Concatenate along the channel dimension (dim=1)
        # Result: (batch_size, 8 * num_sensors, adaptive_pool_cnn_out_size)
        combined_cnn_features = torch.cat(weighted_features_for_lstm, dim=1)
        
        # Permute for LSTM: (batch_size, seq_len, features)
        # Here, seq_len is adaptive_pool_cnn_out_size, features is 8 * num_sensors
        lstm_input = combined_cnn_features.permute(0, 2, 1) # (batch_size, adaptive_pool_cnn_out_size, 8*num_sensors)
        
        # LSTM processing
        # lstm_out: (batch_size, seq_len = adaptive_pool_cnn_out_size, lstm_hidden_size*2)
        lstm_out, _ = self.lstm(lstm_input)
        
        # Apply temporal attention to LSTM output
        # temporal_weights: (batch_size, seq_len = adaptive_pool_cnn_out_size, 1)
        temporal_weights = self.temporal_attention(lstm_out) 
        temporal_weights_softmax = F.softmax(temporal_weights, dim=1)
        # context: (batch_size, lstm_hidden_size*2)
        context = torch.sum(temporal_weights_softmax * lstm_out, dim=1)
        
        # Generate predictions for fault detection
        fault_detection = self.fault_detector(context)
        # Only apply sigmoid in evaluation mode, as training now uses BCEWithLogitsLoss
        if not self.training and not self.mc_dropout:
            fault_detection = torch.sigmoid(fault_detection)
        
        # Process raw_sensor_features_for_anomaly_heads with individual extractors
        sensor_specific_features_for_anomaly_heads = [
            extractor(raw_feat) for extractor, raw_feat in zip(self.sensor_feature_extractors, raw_sensor_features_for_anomaly_heads)
        ]
        
        # Generate individual sensor anomaly scores
        sensor_anomaly_scores = []
        for i in range(self.num_sensors):
            # Concatenate LSTM context with sensor-specific features
            combined_features_for_head = torch.cat([context, sensor_specific_features_for_anomaly_heads[i]], dim=1)
            anomaly_score = self.sensor_anomaly_heads[i](combined_features_for_head)
            sensor_anomaly_scores.append(anomaly_score)
        
        sensor_anomaly_scores_cat = torch.cat(sensor_anomaly_scores, dim=1)  # (batch_size, num_sensors)
        
        return {
            'fault_detection': fault_detection,
            'sensor_anomalies': sensor_anomaly_scores_cat,  # These are NOW LOGITS
            'sensor_attention': sensor_attention_weights, 
            'temporal_attention': temporal_weights_softmax, 
            'joint_anomalies': sensor_anomaly_scores_cat, 
            'traditional_anomalies': sensor_anomaly_scores_cat, 
        }