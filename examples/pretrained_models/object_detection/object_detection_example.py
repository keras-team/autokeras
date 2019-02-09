from autokeras.pretrained import ObjectDetector

if __name__ == '__main__':
    detector = ObjectDetector()
    results = detector.predict("example.jpg", output_file_path="./")

    print(results)
