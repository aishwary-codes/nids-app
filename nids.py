import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
import pyshark
import altair as alt 
import requests
import io
import threading

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = 'data'
MODEL_FILE = os.path.join(DATA_DIR, 'nids_model.pkl')
SCALER_FILE = os.path.join(DATA_DIR, 'nids_scaler.pkl')
LABEL_ENCODERS_FILE = os.path.join(DATA_DIR, 'label_encoders.pkl')
TRAIN_DATA_URL = 'http://kdd.ics.uci.edu/databases/kddcup99/KDDTrain+.txt'
TEST_DATA_URL = 'http://kdd.ics.uci.edu/databases/kddcup99/KDDTest+.txt'
INTERFACE = 'en0'

FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

def download_data(url, filename):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() 
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"{filename} downloaded.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {filename}: {e}. Please check your internet connection or the URL.")
            return None
    return filepath

def packet_summary(packet):
    """Generates a summary string for a given pyshark packet."""
    summary = f"Packet from {packet.highest_layer if hasattr(packet, 'highest_layer') else 'Unknown'}"
    if hasattr(packet, 'ip'):
        summary += f" {packet.ip.src} -> {packet.ip.dst}"
    elif hasattr(packet, 'eth'):
        summary += f" {packet.eth.src} -> {packet.eth.dst}"

    if hasattr(packet, 'tcp'):
        summary += f" TCP:{packet.tcp.srcport}->{packet.tcp.dstport}"
    elif hasattr(packet, 'udp'):
        summary += f" UDP:{packet.udp.srcport}->{packet.udp.dstport}"
    elif hasattr(packet, 'icmp'):
        summary += " ICMP"
    return summary

class NIDS:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = 0
        self.report = {}
        self.train_report = {}
        self.training_results = [] 
        self.feature_names = [f for f in FEATURE_NAMES if f not in ['attack_type', 'difficulty']] 

    def load_data(self):
        st.info("Loading dataset...")
        train_file = download_data(TRAIN_DATA_URL, 'KDDTrain+.txt')
        test_file = download_data(TEST_DATA_URL, 'KDDTest+.txt')

        if not train_file or not test_file:
            return False

        df_train = pd.read_csv(train_file, names=FEATURE_NAMES)
        df_test = pd.read_csv(test_file, names=FEATURE_NAMES)

        combined_df = pd.concat([df_train, df_test], ignore_index=True)
        combined_df = combined_df.drop('difficulty', axis=1)

        combined_df['attack_type'] = combined_df['attack_type'].apply(
            lambda x: 'normal' if x == 'normal' else 'attack'
        )

        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(combined_df[col].unique())
            self.label_encoders[col] = le
            combined_df[col] = le.transform(combined_df[col])

        le_attack = LabelEncoder()
        combined_df['attack_type'] = le_attack.fit_transform(combined_df['attack_type'])
        self.label_encoders['label'] = le_attack 

        self.X_train = combined_df.iloc[:len(df_train)].drop('attack_type', axis=1)
        self.y_train = combined_df.iloc[:len(df_train)]['attack_type']
        self.X_test = combined_df.iloc[len(df_train):].drop('attack_type', axis=1)
        self.y_test = combined_df.iloc[len(df_train):]['attack_type']

        missing_in_train = set(self.feature_names) - set(self.X_train.columns)
        for c in missing_in_train:
            self.X_train[c] = 0
        missing_in_test = set(self.feature_names) - set(self.X_test.columns)
        for c in missing_in_test:
            self.X_test[c] = 0

        self.X_train = self.X_train[self.feature_names]
        self.X_test = self.X_test[self.feature_names]

        self.scaler = StandardScaler()
        numerical_cols = self.X_train.select_dtypes(include=np.number).columns
        self.X_train[numerical_cols] = self.scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = self.scaler.transform(self.X_test[numerical_cols])
        st.success("Dataset loaded and preprocessed.")
        return True

    def train_models(self, progress_bar_obj, chart_placeholder_obj):
        if self.X_train is None or self.y_train is None:
            st.error("Data not loaded or preprocessed. Cannot train models.")
            return "N/A", 0.0

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        self.training_results = []
        best_accuracy = 0
        best_model_name = ""
        best_model_object = None

        chart_data = pd.DataFrame(columns=['Model', 'Accuracy Type', 'Accuracy'])
            
        for i, (name, model) in enumerate(models.items()):
            progress_bar_obj.progress((i + 1) / len(models), text=f"Training: {name}...")
                
            model.fit(self.X_train, self.y_train)
                
            y_pred_train = model.predict(self.X_train)
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
                
            y_pred_test = model.predict(self.X_test)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)

            self.training_results.append({
                'Model': name,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Model Object': model
            })
            
            chart_data = pd.concat([chart_data, pd.DataFrame([
                {'Model': name, 'Accuracy Type': 'Train Accuracy', 'Accuracy': train_accuracy},
                {'Model': name, 'Accuracy Type': 'Test Accuracy', 'Accuracy': test_accuracy}
            ])], ignore_index=True)

            with chart_placeholder_obj.container(): 
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Model:N', sort=list(models.keys())), 
                    y='Accuracy:Q',
                    color='Accuracy Type:N',
                    tooltip=['Model', 'Accuracy Type', alt.Tooltip('Accuracy', format='.4f')]
                ).properties(
                    title='Model Training Accuracy Comparison'
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_name = name
                best_model_object = model 

        self.model = best_model_object 
        self.accuracy = best_accuracy
        
        if self.model:
            y_pred_best_test = self.model.predict(self.X_test)
            self.report = classification_report(self.y_test, y_pred_best_test, target_names=['normal', 'attack'], output_dict=True)
            
            y_pred_best_train = self.model.predict(self.X_train)
            self.train_report = classification_report(self.y_train, y_pred_best_train, target_names=['normal', 'attack'], output_dict=True)

        return best_model_name, best_accuracy

    def save_model(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        try:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(LABEL_ENCODERS_FILE, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            st.success("Model, scaler, and label encoders saved.")
        except Exception as e:
            st.error(f"Error saving model: {e}")

    def load_model(self):
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(LABEL_ENCODERS_FILE):
                with open(MODEL_FILE, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_FILE, 'rb') as f:
                    self.scaler = pickle.load(f)
                with open(LABEL_ENCODERS_FILE, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                return True
            return False
        except Exception as e:
            st.error(f"Error loading models: {e}. Model files might be corrupted or incompatible.")
            return False

    def load_or_train_model(self, force_retrain=False, progress_bar_obj=None, chart_placeholder_obj=None):
        if force_retrain or not self.load_model():
            st.info("Training new model...")
            if not self.load_data(): 
                st.error("Failed to load data for training. Aborting model training.")
                return

            best_model_name, best_accuracy = self.train_models(progress_bar_obj, chart_placeholder_obj)
            if best_model_name != "N/A":
                self.save_model()
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model_accuracy = best_accuracy
                st.session_state.training_results = self.training_results
                st.session_state.final_report = self.report
                st.session_state.final_train_report = self.train_report
            else:
                st.error("Model training failed.")
        else:
            st.info("Model loaded from disk.")
            if self.X_test is None or self.y_test is None:
                self.load_data()
            
            if self.model and self.X_test is not None and self.y_test is not None:
                st.session_state.best_model_name = self.model.__class__.__name__
                y_pred_test = self.model.predict(self.X_test)
                self.accuracy = accuracy_score(self.y_test, y_pred_test)
                self.report = classification_report(self.y_test, y_pred_test, target_names=['normal', 'attack'], output_dict=True)

                y_pred_train = self.model.predict(self.X_train)
                self.train_report = classification_report(self.y_train, y_pred_train, target_names=['normal', 'attack'], output_dict=True)
                
                st.session_state.best_model_accuracy = self.accuracy
                st.session_state.final_report = self.report
                st.session_state.final_train_report = self.train_report
            
            if 'training_results' not in st.session_state or not st.session_state.training_results:
                pass 

    def preprocess_packet(self, packet):
        features = {name: 0.0 if name not in ['protocol_type', 'service', 'flag'] else 'other' for name in self.feature_names}
        
        features['duration'] = float(packet.sniff_timestamp) if hasattr(packet, 'sniff_timestamp') else 0.0
        features['src_bytes'] = int(packet.length) if hasattr(packet, 'length') else 0
        features['dst_bytes'] = 0 

        if hasattr(packet, 'ip'):
            protocol_num = getattr(packet.ip, 'proto', 'unknown')
            if protocol_num == '6': features['protocol_type'] = 'tcp'
            elif protocol_num == '17': features['protocol_type'] = 'udp'
            elif protocol_num == '1': features['protocol_type'] = 'icmp'
            else: features['protocol_type'] = 'other'
        elif hasattr(packet, 'eth'):
            if hasattr(packet.eth, 'type'):
                eth_type = getattr(packet.eth, 'type', 'unknown')
                if eth_type == '0x0806': features['protocol_type'] = 'arp' 
            else:
                 features['protocol_type'] = 'other' 
       
        if hasattr(packet, 'tcp'):
            dst_port = getattr(packet.tcp, 'dstport', None)
            src_port = getattr(packet.tcp, 'srcport', None)
            if dst_port == '80' or src_port == '80': features['service'] = 'http'
            elif dst_port == '443' or src_port == '443': features['service'] = 'http'
            elif dst_port == '21' or src_port == '21': features['service'] = 'ftp_data' 
            elif dst_port == '23' or src_port == '23': features['service'] = 'telnet'
            else: features['service'] = 'other'

            flags = ''
            if hasattr(packet.tcp, 'flags_ack') and packet.tcp.flags_ack == '1': flags += 'A'
            if hasattr(packet.tcp, 'flags_fin') and packet.tcp.flags_fin == '1': flags += 'F'
            if hasattr(packet.tcp, 'flags_push') and packet.tcp.flags_push == '1': flags += 'P'
            if hasattr(packet.tcp, 'flags_reset') and packet.tcp.flags_reset == '1': flags += 'R'
            if hasattr(packet.tcp, 'flags_syn') and packet.tcp.flags_syn == '1': flags += 'S'
            if hasattr(packet.tcp, 'flags_urg') and packet.tcp.flags_urg == '1': flags += 'U'
            features['flag'] = flags if flags else 'S0'

        elif hasattr(packet, 'udp'):
            dst_port = getattr(packet.udp, 'dstport', None)
            src_port = getattr(packet.udp, 'srcport', None)
            if dst_port == '53' or src_port == '53': features['service'] = 'dns'
            else: features['service'] = 'other'
            features['flag'] = '0' 
        elif hasattr(packet, 'icmp'):
            features['service'] = 'icmp'
            features['flag'] = '0'
        else: 
            features['service'] = 'other'
            features['flag'] = '0'

        df_features = pd.DataFrame([features])

        for col in ['protocol_type', 'service', 'flag']:
            if col in self.label_encoders and hasattr(self.label_encoders[col], 'classes_'):
                val_to_encode = str(df_features[col].iloc[0])
                if val_to_encode not in self.label_encoders[col].classes_:
                    if 'other' in self.label_encoders[col].classes_:
                        df_features[col] = self.label_encoders[col].transform(['other'])[0]
                    else:
                        df_features[col] = 0
                else:
                    df_features[col] = self.label_encoders[col].transform([val_to_encode])[0]
            else:
                df_features[col] = 0 

        expected_features = [f for f in FEATURE_NAMES if f not in ['attack_type', 'difficulty']]
        
        for f in expected_features:
            if f not in df_features.columns:
                df_features[f] = 0

        df_features = df_features[expected_features]

        numerical_cols = df_features.select_dtypes(include=np.number).columns
        if not numerical_cols.empty and self.scaler:
            df_features[numerical_cols] = self.scaler.transform(df_features[numerical_cols])

        return df_features

    def analyze_packets_batch(self, packets):
        analyzed_results = []
        if not self.model or not self.scaler or not self.label_encoders:
            return []

        for packet in packets:
            try:
                processed_features = self.preprocess_packet(packet)
                prediction = self.model.predict(processed_features)
                prediction_proba = self.model.predict_proba(processed_features)

                decoded_prediction = self.label_encoders['label'].inverse_transform(prediction)[0]
                
                class_labels = self.label_encoders['label'].classes_
                confidence_attack = 0.0
                confidence_normal = 0.0
                
                if 'attack' in class_labels and 'normal' in class_labels:
                    attack_idx = np.where(class_labels == 'attack')[0][0]
                    normal_idx = np.where(class_labels == 'normal')[0][0]
                    
                    confidence_attack = prediction_proba[0][attack_idx]
                    confidence_normal = prediction_proba[0][normal_idx]
                else:
                    confidence_attack = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else 0.0
                    confidence_normal = prediction_proba[0][0] if len(prediction_proba[0]) > 0 else 0.0


                analyzed_results.append({
                    'raw': packet,
                    'timestamp': float(packet.sniff_timestamp),
                    'source_ip': packet.ip.src if hasattr(packet, 'ip') else 'N/A',
                    'destination_ip': packet.ip.dst if hasattr(packet, 'ip') else 'N/A',
                    'protocol_layer': packet.highest_layer if hasattr(packet, 'highest_layer') else 'N/A',
                    'length': packet.length if hasattr(packet, 'length') else 'N/A',
                    'prediction': decoded_prediction,
                    'confidence_attack': confidence_attack,
                    'confidence_normal': confidence_normal
                })
            except Exception as e:
                analyzed_results.append({
                    'raw': packet,
                    'timestamp': float(packet.sniff_timestamp),
                    'source_ip': packet.ip.src if hasattr(packet, 'ip') else 'N/A',
                    'destination_ip': packet.ip.dst if hasattr(packet, 'ip') else 'N/A',
                    'protocol_layer': packet.highest_layer if hasattr(packet, 'highest_layer') else 'N/A',
                    'length': packet.length if hasattr(packet, 'length') else 'N/A',
                    'prediction': 'Processing Error',
                    'confidence_attack': 0.0,
                    'confidence_normal': 0.0
                })
        return analyzed_results

def run_dashboard():
    st.set_page_config(page_title="Network Intrusion Detection System", layout="wide", initial_sidebar_state="expanded")
    st.title("🔐 Network Intrusion Detection System Dashboard")
    st.markdown("### Real-time monitoring and detection using NSL-KDD and ML models")

    if 'nids_instance' not in st.session_state:
        st.session_state.nids_instance = NIDS()
    nids = st.session_state.nids_instance

    if 'capture_active' not in st.session_state:
        st.session_state.capture_active = False
    if 'analyzed_packets_display' not in st.session_state:
        st.session_state.analyzed_packets_display = pd.DataFrame(columns=[
            "Timestamp", "Source", "Destination", "Protocol", "Length", "Prediction", "Confidence (Attack)", "Summary"
        ])
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = "N/A"
    if 'best_model_accuracy' not in st.session_state:
        st.session_state.best_model_accuracy = 0.0
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'final_report' not in st.session_state:
        st.session_state.final_report = {}
    if 'final_train_report' not in st.session_state:
        st.session_state.final_train_report = {}
    if 'packet_capture_thread' not in st.session_state:
        st.session_state.packet_capture_thread = None
    if 'stop_capture_event' not in st.session_state:
        st.session_state.stop_capture_event = threading.Event()

    with st.sidebar:
        st.header("⚙️ Controls")

        st.subheader("Model Management")
        retrain_button = st.button("🔄 Retrain Model", key="retrain_button")
        
        st.subheader("Live Packet Capture")
        capture_count = st.slider("Number of packets per cycle", min_value=1, max_value=50, value=10)
        refresh_interval = st.slider("Refresh interval (seconds)", min_value=1, max_value=10, value=2)

        if st.button("▶️ Start Continuous Capture" if not st.session_state.capture_active else "⏹️ Stop Capture", key="toggle_capture_button"):
            if not st.session_state.capture_active:
                if nids.model and nids.scaler and nids.label_encoders:
                    st.session_state.capture_active = True
                    st.session_state.stop_capture_event.clear() 
                    
                    st.session_state.packet_capture_thread = threading.Thread(
                        target=capture_loop,
                        args=(nids, capture_count, refresh_interval, st.session_state.stop_capture_event)
                    )
                    st.session_state.packet_capture_thread.daemon = True 
                    st.session_state.packet_capture_thread.start()
                    st.success("Packet capture started. Scroll down to see live data.")
                else:
                    st.warning("Model is not trained or loaded. Please train the model first to start live capture.")
                    st.session_state.capture_active = False 
            else: 
                st.session_state.stop_capture_event.set() 
                st.session_state.capture_active = False
                st.info("Stopping packet capture...")
                st.success("Packet capture stopped.")
            st.rerun() 

    st.header("📊 Model Training & Performance")
    
    training_status_text = st.empty()
    training_progress_bar = st.empty() 
    accuracy_chart_placeholder = st.empty()

    if retrain_button:
        training_status_text.info("Training new model. This may take some time...")
        st.session_state.training_results = []
        st.session_state.best_model_name = "N/A"
        st.session_state.best_model_accuracy = 0.0
        st.session_state.final_report = {}
        st.session_state.final_train_report = {}
        
        nids.load_or_train_model(
            force_retrain=True,
            progress_bar_obj=training_progress_bar.progress(0),
            chart_placeholder_obj=accuracy_chart_placeholder
        )
        training_status_text.success("Model training complete!")
        training_progress_bar.empty() 
        accuracy_chart_placeholder.empty() 
        
    else: 
        if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(LABEL_ENCODERS_FILE)):
            training_status_text.info("Model files not found. Training model for the first time...")
            nids.load_or_train_model(
                force_retrain=True,
                progress_bar_obj=training_progress_bar.progress(0),
                chart_placeholder_obj=accuracy_chart_placeholder
            )
            training_status_text.success("Model training complete!")
            training_progress_bar.empty()
            accuracy_chart_placeholder.empty()
        else:
            training_status_text.info("Model loaded from disk.")
            nids.load_or_train_model(
                force_retrain=False,
                progress_bar_obj=training_progress_bar, 
                chart_placeholder_obj=accuracy_chart_placeholder
            )
            training_progress_bar.empty() 
            accuracy_chart_placeholder.empty() 

    if st.session_state.training_results:
        st.subheader("Individual Model Performance")
        df_results_raw = pd.DataFrame(st.session_state.training_results)
        if 'Model Object' in df_results_raw.columns:
            df_results_for_chart = df_results_raw.drop(columns=['Model Object'])
        else:
            df_results_for_chart = df_results_raw

        df_results_melted = df_results_for_chart.melt(id_vars=['Model'], var_name='Accuracy Type', value_name='Accuracy')

        accuracy_chart = alt.Chart(df_results_melted).mark_bar().encode(
            x=alt.X('Model:N', sort=alt.EncodingSortField(field="Accuracy", op="max", order='descending')),
            y='Accuracy:Q',
            color='Accuracy Type:N',
            tooltip=['Model', 'Accuracy Type', alt.Tooltip('Accuracy', format='.4f')]
        ).properties(
            title='Model Accuracy Comparison'
        ).interactive()
        st.altair_chart(accuracy_chart, use_container_width=True)

    if st.session_state.best_model_name != "N/A":
        st.markdown(f"### 🏆 **Best Model: {st.session_state.best_model_name}** with Test Accuracy: **{st.session_state.best_model_accuracy:.4f}**")

        st.subheader("Classification Report (Test Data) - Best Model")
        if st.session_state.final_report:
            df_report_test = pd.DataFrame(st.session_state.final_report).transpose()
            st.dataframe(df_report_test.style.format("{:.4f}"))
        else:
            st.info("No classification report available for test data.")

        st.subheader("Classification Report (Train Data) - Best Model")
        if st.session_state.final_train_report:
            df_report_train = pd.DataFrame(st.session_state.final_train_report).transpose()
            st.dataframe(df_report_train.style.format("{:.4f}"))
        else:
            st.info("No classification report available for train data.")

    st.markdown("---")

    st.header("📡 Live Packet Prediction")

    live_packet_status = st.empty()
    packet_table_placeholder = st.empty()
    alert_metrics_placeholder = st.empty()

    if not st.session_state.capture_active:
        live_packet_status.info("Click 'Start Continuous Capture' to begin monitoring.")
        with packet_table_placeholder:
            if not st.session_state.analyzed_packets_display.empty:
                st.dataframe(st.session_state.analyzed_packets_display.set_index("Timestamp"), use_container_width=True)
            else:
                st.info("No packets captured yet.")
        with alert_metrics_placeholder:
            attack_count = st.session_state.analyzed_packets_display[st.session_state.analyzed_packets_display['Prediction'] == 'attack'].shape[0]
            total_analyzed = len(st.session_state.analyzed_packets_display)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Packets Analyzed", value=total_analyzed)
            with col2:
                if attack_count > 0:
                    st.error(f"🚨 **{attack_count} Intrusion(s) Detected!**")
                else:
                    st.success("✓ No Intrusion Detected")

def capture_loop(nids_instance, count_per_cycle, refresh_interval, stop_event):
    """
    Function to run packet capture and analysis in a separate thread.
    Updates Streamlit's session_state, which will trigger UI updates on next rerun.
    """
    if not nids_instance.model or not nids_instance.scaler or not nids_instance.label_encoders:
        st.session_state.capture_active = False 
        return

    st.session_state.analyzed_packets_display = pd.DataFrame(columns=[
        "Timestamp", "Source", "Destination", "Protocol", "Length", "Prediction", "Confidence (Attack)", "Summary"
    ]) 

    try:
        capture = pyshark.LiveCapture(interface=INTERFACE)
        capture.set_debug()

        st.session_state.live_capture_status = f"Capturing packets from interface '{INTERFACE}'..."
        
        for packet in capture.sniff_continuously():
            if stop_event.is_set():
                break 

            analyzed_pkt = nids_instance.analyze_packets_batch([packet])
            
            if analyzed_pkt:
                pkt = analyzed_pkt[0] 
                new_row = {
                    "Timestamp": pd.to_datetime(pkt['timestamp'], unit='s'),
                    "Source": pkt['source_ip'],
                    "Destination": pkt['destination_ip'],
                    "Protocol": pkt['protocol_layer'],
                    "Length": pkt['length'],
                    "Prediction": pkt['prediction'],
                    "Confidence (Attack)": f"{pkt.get('confidence_attack', 0):.2f}",
                    "Summary": packet_summary(pkt['raw'])
                }

                st.session_state.analyzed_packets_display = pd.concat(
                    [st.session_state.analyzed_packets_display, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                max_display_rows = 100
                if len(st.session_state.analyzed_packets_display) > max_display_rows:
                    st.session_state.analyzed_packets_display = st.session_state.analyzed_packets_display.tail(max_display_rows).reset_index(drop=True)

    except pyshark.capture.capture.TSharkNotFoundException:
        st.session_state.live_capture_status = "TShark not found. Ensure Wireshark (and TShark) is installed and in your system's PATH."
        st.session_state.capture_active = False
    except Exception as e:
        st.session_state.live_capture_status = f"An error occurred during packet capture: {e}"
        st.session_state.capture_active = False
    finally:
        if 'capture' in locals() and capture.running:
            capture.close()
        st.session_state.live_capture_status = "Packet capture stopped."
        st.session_state.capture_active = False 


if __name__ == '__main__':
    if 'packet_table_placeholder' not in st.session_state:
        st.session_state.packet_table_placeholder = st.empty()
    if 'alert_metrics_placeholder' not in st.session_state:
        st.session_state.alert_metrics_placeholder = st.empty()
    if 'live_packet_status_placeholder' not in st.session_state:
        st.session_state.live_packet_status_placeholder = st.empty()
    
    run_dashboard()

    if st.session_state.capture_active:
        st.session_state.live_packet_status_placeholder.info(st.session_state.get('live_capture_status', "Capturing packets..."))

        with st.session_state.packet_table_placeholder.container():
            if not st.session_state.analyzed_packets_display.empty:
                st.dataframe(st.session_state.analyzed_packets_display.set_index("Timestamp"), use_container_width=True)
            else:
                st.info("No packets captured yet. Waiting for traffic...")
        
        with st.session_state.alert_metrics_placeholder.container():
            attack_count = st.session_state.analyzed_packets_display[st.session_state.analyzed_packets_display['Prediction'] == 'attack'].shape[0]
            total_analyzed = len(st.session_state.analyzed_packets_display)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Packets Analyzed", value=total_analyzed)
            with col2:
                if attack_count > 0:
                    st.error(f"🚨 **{attack_count} Intrusion(s) Detected!**")
                else:
                    st.success("✓ No Intrusion Detected")
        
        time.sleep(st.session_state.get('refresh_interval', 2))
        st.rerun()