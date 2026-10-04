# Component Orientation Detection

Computer vision based inspection system for checking the orientation of components on a fixture.

## Technologies
- Python
- OpenCV
- NumPy
- Tkinter
- Pillow

## Usage

```text
pip install -r requirements.txt
python main.py
```

The application supports image upload and camera input. It aligns the fixture, detects the components and evaluates lever orientation.

## Files
- `main.py` - application interface, camera, image upload and results
- `registration.py` - image registration and alignment
- `detector.py` - component detection and inspection
- `lever_detector.py` - lever shape and orientation detection
- `visualization.py` - result visualization
- `config.py` - detection settings
- `reference.jpg` - known-good reference image
