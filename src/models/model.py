import torch
import torch.nn as nn
import torch.nn.functional as F

class GearboxCNNLSTM(nn.Module):
    def __init__(self, input_channels=9, window_size=256, lstm_hidden_size=32, num_lstm_layers=1, num_sensors=8, dropout_rate=0.3):
        super().__init__()
        self.num_samples = 0
        self.num_sensors = num_sensors
        self.dropout_rate = dropout_rate
        self.mc_dropout = False  # Flag for Monte Carlo dropout during inference
        self.lstm_hidden_size = lstm_hidden_size
        
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
        
        # RPM processing branch
        self.rpm_cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # Add dropout to RPM processing
            nn.MaxPool1d(4)
        )
        
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
        self.lstm = nn.LSTM(
            input_size=8 * (num_sensors + 1),  # Combined features from all sensors and RPM
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
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
        
        # Cross-sensor interaction layer to model dependencies between sensors
        self.cross_sensor_interaction = nn.Sequential(
            nn.Linear(num_sensors * 8 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )
        
        # Individual sensor feature extractors
        self.sensor_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(8 * 16, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2)
            ) for _ in range(num_sensors)
        ])
        
        # Improved sensor anomaly detection heads with cross-sensor attention
        self.sensor_anomaly_heads = nn.ModuleList([
            nn.Sequential(
                # Input: Global context + sensor-specific features + global sensor interaction features
                nn.Linear(lstm_hidden_size*2 + 32 + 64, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(num_sensors)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
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
        
        # Split input into sensor data and RPM
        sensor_data = x[:, :, :self.num_sensors]  # (batch_size, time_steps, num_sensors)
        rpm_data = x[:, :, -1:]  # (batch_size, time_steps, 1)
        
        # Process each sensor independently
        sensor_features = []
        sensor_attentions = []
        raw_sensor_features = []  # Store raw sensor features for anomaly detection
        
        for i in range(self.num_sensors):
            # Extract single sensor data and reshape
            single_sensor = sensor_data[:, :, i:i+1].permute(0, 2, 1)  # (batch_size, 1, time_steps)
            
            # Apply sensor-specific CNN
            sensor_feat = self.sensor_cnns[i](single_sensor)
            sensor_feat = self.adaptive_pool(sensor_feat)  # (batch_size, 8, 16)
            
            # Store raw features for later use in anomaly detection
            flat_feat = sensor_feat.view(batch_size, -1)
            raw_sensor_features.append(flat_feat)
            
            # Calculate attention for this sensor
            attention = self.sensor_attention(flat_feat)
            sensor_attentions.append(attention)
            
            sensor_features.append(sensor_feat)
        
        # Process RPM data
        rpm = rpm_data.permute(0, 2, 1)  # (batch_size, 1, time_steps)
        rpm_features = self.rpm_cnn(rpm)
        rpm_features = self.adaptive_pool(rpm_features)
        
        # Combine all features with attention
        sensor_attentions = F.softmax(torch.cat(sensor_attentions, dim=1), dim=1)
        
        # Reshape and combine features
        all_features = []
        for i, feat in enumerate(sensor_features):
            weighted_feat = feat * sensor_attentions[:, i:i+1].unsqueeze(-1)
            all_features.append(weighted_feat)
        all_features.append(rpm_features)
        
        combined = torch.cat(all_features, dim=1)  # (batch_size, features, time_steps)
        combined = combined.permute(0, 2, 1)  # (batch_size, time_steps, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        
        # Apply temporal attention
        temporal_weights = self.temporal_attention(lstm_out)
        temporal_weights = F.softmax(temporal_weights, dim=1)
        context = torch.sum(temporal_weights * lstm_out, dim=1)
        
        # Generate predictions for fault detection
        fault_detection = self.fault_detector(context)
        # Only apply sigmoid in evaluation mode, as training now uses BCEWithLogitsLoss
        if not self.training and not self.mc_dropout:
            fault_detection = torch.sigmoid(fault_detection)
        
        # Process raw sensor features with individual extractors
        sensor_specific_features = [
            extractor(raw_feat) for extractor, raw_feat in zip(self.sensor_feature_extractors, raw_sensor_features)
        ]
        
        # Cross-sensor interaction - concatenate all raw sensor features and extract global patterns
        all_sensor_features = torch.cat(raw_sensor_features, dim=1)
        global_sensor_interaction = self.cross_sensor_interaction(all_sensor_features)
        
        # Generate sensor-wise anomaly scores with more independence
        sensor_anomalies = []
        for i in range(self.num_sensors):
            # Combine global context with sensor-specific features and cross-sensor information
            combined_features = torch.cat([context, sensor_specific_features[i], global_sensor_interaction], dim=1)
            
            # Generate anomaly score through the sensor-specific head
            anomaly_score = self.sensor_anomaly_heads[i](combined_features)
            sensor_anomalies.append(anomaly_score)
        
        # Concatenate anomaly scores for all sensors
        sensor_anomalies = torch.cat(sensor_anomalies, dim=1)  # (batch_size, num_sensors)
        
        return {
            'fault_detection': fault_detection,
            'sensor_anomalies': sensor_anomalies,  # Anomaly score for each sensor
            'sensor_attention': sensor_attentions,  # How much attention each sensor gets
            'temporal_attention': temporal_weights  # How important each timestep is
        }