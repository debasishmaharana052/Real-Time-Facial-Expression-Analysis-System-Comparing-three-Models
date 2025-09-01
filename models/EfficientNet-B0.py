# real_time_emotion_detection_with_plots.py
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pandas as pd


class RealTimeEmotionDetectorWithPlots:
    def __init__(self, model_path=r"C:\Users\LENOVO\OneDrive\Desktop\Debasish\FER Project\efficientnet_b0_best.pth"):
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Data collection for plotting
        self.emotion_counts = defaultdict(int)
        self.confidence_history = deque(maxlen=100)  # Store last 100 predictions
        self.emotion_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=50)
        self.session_data = []
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("🎯 Model loaded successfully!")
        print("📊 Test Accuracy: 67.12% (EfficientNet-B0)")
        print(f"💻 Running on: {self.device}")
    
    def predict_emotion(self, face_image):
        """Predict emotion and log data"""
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return "Unknown", 0.0
            
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
            
            emotion = self.emotion_labels[predicted.item()]
            conf_val = confidence.item()
            
            # Log data for plotting
            self.emotion_counts[emotion] += 1
            self.confidence_history.append(conf_val)
            self.emotion_history.append(emotion)
            self.session_data.append({
                'timestamp': time.time(),
                'emotion': emotion,
                'confidence': conf_val
            })
            
            return emotion, conf_val
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0
    
    def run_real_time(self):
        """Real-time detection with data collection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam")
            return
        
        print("\n🎥 Real-Time Facial Expression Recognition with Plotting!")
        print("📋 Controls:")
        print("   'q' - Quit and show plots")
        print("   's' - Save screenshot")
        print("   'p' - Show current plots")
        print("   'f' - Toggle FPS display")
        
        fps_counter = 0
        current_fps = 0.0
        start_time = time.time()
        show_fps = True
        confidence_threshold = 0.6
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence = self.predict_emotion(face_roi)
                
                color = (0, 255, 0) if confidence > confidence_threshold else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{emotion}: {confidence:.1%}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Info overlay
            cv2.putText(frame, "Debasish's FER Project - EfficientNet-B0 | 67.12% Accuracy", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS calculation and display
            if show_fps:
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = 30 / elapsed if elapsed > 0 else 0
                    self.fps_history.append(current_fps)
                    start_time = time.time()
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Data info
            total_detections = sum(self.emotion_counts.values())
            cv2.putText(frame, f"Total Detections: {total_detections}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(frame, "q:quit+plots | s:save | p:plots | f:fps", 
                       (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Debasish FER Project - Real-Time with Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'emotion_capture_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('p'):
                self.show_current_plots()
            elif key == ord('f'):
                show_fps = not show_fps
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final plots
        self.generate_final_plots()
        print("🏁 Real-time detection ended with plots generated!")
    
    def show_current_plots(self):
        """Show current statistics during runtime"""
        if not self.emotion_counts:
            print("No data collected yet!")
            return
        
        # Quick stats
        total = sum(self.emotion_counts.values())
        print(f"\n📊 Current Statistics (Total: {total} detections):")
        for emotion in self.emotion_labels:
            count = self.emotion_counts[emotion]
            percentage = (count/total)*100 if total > 0 else 0
            print(f"   {emotion}: {count} ({percentage:.1f}%)")
        
        if self.confidence_history:
            avg_confidence = np.mean(list(self.confidence_history))
            print(f"   Average Confidence: {avg_confidence:.1%}")
    
    def generate_final_plots(self):
        """Generate comprehensive plots after session"""
        if not self.emotion_counts:
            print("No data to plot!")
            return
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Debasish\'s Real-Time Facial Expression Recognition Results', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust subplot spacing
        plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.90, 
                           wspace=0.25, hspace=0.35)
        
        # 1. Emotion Distribution Bar Chart
        ax1 = plt.subplot(2, 3, 1)
        emotions = list(self.emotion_counts.keys())
        counts = list(self.emotion_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
        bars = ax1.bar(emotions, counts, color=colors)
        ax1.set_title('Detected Emotion Counts', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Emotions', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.tick_params(axis='y', labelsize=9)
        
        # Add count labels on bars with better positioning
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Emotion Distribution Pie Chart
        ax2 = plt.subplot(2, 3, 2)
        wedges, texts, autotexts = ax2.pie(counts, labels=emotions, autopct='%1.1f%%', 
                                          startangle=90, colors=colors)
        ax2.set_title('Emotion Distribution (%)', fontsize=12, fontweight='bold')
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        for text in texts:
            text.set_fontsize(9)
        
        # 3. Confidence Over Time
        ax3 = plt.subplot(2, 3, 3)
        if len(self.confidence_history) > 0:
            ax3.plot(list(self.confidence_history), color='green', linewidth=1.5)
            ax3.set_title('Confidence Over Time', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Detection Number', fontsize=10)
            ax3.set_ylabel('Confidence', fontsize=10)
            ax3.axhline(y=0.6, color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label='Threshold')
            ax3.legend(fontsize=9)
            ax3.tick_params(labelsize=9)
            ax3.grid(True, alpha=0.3)
        
        # 4. FPS Performance
        ax4 = plt.subplot(2, 3, 4)
        if len(self.fps_history) > 0:
            ax4.plot(list(self.fps_history), color='orange', linewidth=2, marker='o', markersize=3)
            ax4.set_title('FPS Performance', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time Interval', fontsize=10)
            ax4.set_ylabel('FPS', fontsize=10)
            avg_fps = np.mean(list(self.fps_history))
            ax4.axhline(y=avg_fps, color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label=f'Average: {avg_fps:.1f}')
            ax4.legend(fontsize=9)
            ax4.tick_params(labelsize=9)
            ax4.grid(True, alpha=0.3)
        
        # 5. Confidence Distribution
        ax5 = plt.subplot(2, 3, 5)
        if len(self.confidence_history) > 0:
            n, bins, patches = ax5.hist(list(self.confidence_history), bins=15, 
                                       alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
            ax5.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Confidence Level', fontsize=10)
            ax5.set_ylabel('Frequency', fontsize=10)
            ax5.axvline(x=0.6, color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label='Threshold')
            ax5.legend(fontsize=9)
            ax5.tick_params(labelsize=9)
            ax5.grid(True, alpha=0.3)
        
        # 6. Performance Summary Text (Improved Layout)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate statistics
        total_detections = sum(counts)
        avg_confidence = np.mean(list(self.confidence_history)) if self.confidence_history else 0
        avg_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
        high_conf_ratio = sum(1 for c in self.confidence_history if c > 0.6) / len(self.confidence_history) if self.confidence_history else 0
        
        # Create clean, well-spaced summary text
        summary_lines = [
            "SESSION SUMMARY",
            "─" * 25,
            f"Total Detections: {total_detections}",
            f"Average Confidence: {avg_confidence:.1%}",
            f"Average FPS: {avg_fps:.1f}",
            f"High Confidence Rate: {high_conf_ratio:.1%}",
            "",
            "MODEL PERFORMANCE", 
            "─" * 25,
            "Architecture: EfficientNet-B0",
            "Test Accuracy: 67.12%",
            f"Device: {self.device.upper()}"
        ]
        
        # Position text with proper line spacing
        y_start = 0.95
        line_height = 0.07
        
        for i, line in enumerate(summary_lines):
            y_pos = y_start - (i * line_height)
            fontweight = 'bold' if line in ['SESSION SUMMARY', 'MODEL PERFORMANCE'] or '─' in line else 'normal'
            fontsize = 11 if fontweight == 'bold' else 10
            
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes, 
                    fontsize=fontsize, fontweight=fontweight,
                    verticalalignment='top', fontfamily='monospace')
        
        # Save plot with high quality
        filename = f'fer_results_{int(time.time())}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Plots saved as: {filename}")
        
        plt.show()
        
        # Save data to CSV
        if self.session_data:
            df = pd.DataFrame(self.session_data)
            csv_filename = f'fer_session_data_{int(time.time())}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"💾 Session data saved as: {csv_filename}")


# Run the enhanced application
if __name__ == "__main__":
    detector = RealTimeEmotionDetectorWithPlots()
    detector.run_real_time()
