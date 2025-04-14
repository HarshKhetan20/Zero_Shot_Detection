# 🎥 Zero-Shot Object Detection with OWL-ViT

An easy-to-use tool for **real-time object detection** on video files using the state-of-the-art **OWL-ViT** model. This project performs zero-shot object detection and logs the results, all while displaying the video with bounding boxes around detected objects.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## ✨ Features

- **Zero-Shot Detection**: No need for training; uses the OWL-ViT model for object detection.
- **Real-Time Detection**: Detects and displays objects in real time as the video plays.
- **Automatic Logging**: Logs each detection (object type and confidence) in a text file in real time.
- **Interactive Video Display**: Shows video frames with bounding boxes drawn around detected objects.
- **Customizable Prompts**: Easily add or change the list of object classes to detect.

---

## 🛠️ Installation

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/your-username/zero-shot-object-detection.git
    cd zero-shot-object-detection
    ```

2. **Install dependencies**:

    You can install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

   Or manually:

    ```bash
    pip install torch transformers opencv-python Pillow
    ```

---

## 🚀 Usage

1. **Run the Python Script**:

    After installation, run the script to start detecting objects in a video:

    ```bash
    python object_detection.py
    ```

2. **Select a Video**:

    The script will prompt you to select a video file (MP4 or any supported format). The video will then be processed frame by frame.

3. **Watch the Detection in Action**:

    The video will be displayed with bounding boxes around detected objects, and the corresponding object names and confidence scores will be shown.

4. **Check the Log File**:

    All detected objects, along with their confidence scores, are logged in a text file `detection_log.txt` in the same folder as the video.

    Example log entry:

    ```
    Detected: a lightbulb with confidence: 0.92
    Detected: a lion with confidence: 0.85
    ```

---

## 🔧 Customization

You can easily modify the detection behavior:

- **Change Detection Prompts**: The list of objects to detect is customizable. Modify the `prompts` list in the code:

    ```python
    prompts = ["a lightbulb", "a matchstick", "a monitor", "a lion", "a gaming console"]
    ```

- **Adjust Confidence Threshold**: Modify the threshold for detection in the code to control the minimum confidence for logging an object:

    ```python
    results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_size, threshold=0.1)[0]
    ```

- **Logging**: The script logs results in `detection_log.txt`. You can change the log filename or format if needed.

---

## 📁 File Structure

zero-shot-object-detection/

│

├── object_detection.py        # Main script for object detection

├── detection_log.txt          # Log file for detections (generated in real time)

├── requirements.txt           # List of required Python libraries

└── README.md                  # This README file

---


---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 💬 Contributing

I welcome contributions! If you'd like to improve the project, feel free to fork the repository and submit a pull request. For more information, check out the [contributing guidelines](CONTRIBUTING.md).

---

## 📚 Resources

- [OWL-ViT on Hugging Face](https://huggingface.co/google/owlvit-base-patch32)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

## 👨‍💻 Author

Created by [Harsh Khetan](https://github.com/HarshKhetan20). 

Feel free to reach out for any questions or improvements!



