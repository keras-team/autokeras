from autokeras.pretrained.object_detector import ObjectDetector

if __name__ == '__main__':
    detector = ObjectDetector()
    detector.load()
    results = detector.predict("example.jpg", output_file_path="./")

    print(results)
