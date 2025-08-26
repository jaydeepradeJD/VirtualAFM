# 🔬 Virtual AFM Gradio Web Interface

A user-friendly web interface for the Virtual Atomic Force Microscope (AFM) simulation using Gradio.

## Features

- **📁 Easy File Upload**: Simply drag and drop your OBJ files
- **🎛️ Interactive Controls**: Adjust elevation, azimuthal, and roll angles in real-time
- **⚙️ Customizable AFM Parameters**: Control tip radius and scan step size
- **📊 Dual Output**: View both the AFM simulation and raw depth map
- **💻 Web-based**: No complex setup - runs in your browser

## Quick Start

### Option 1: Simple Launch
```bash
# Make sure you have the environment activated
conda activate virtual_afm  # or your environment name

# Install Gradio if not already installed
pip install gradio

# Launch the app
python gradio_virtual_afm.py
```

### Option 2: Using the Launcher
```bash
# Use the provided launcher script
python launch_gradio.py
```

The app will start and be available at `http://localhost:7860`

## Usage Guide

### 1. Upload Your OBJ File
- Click "Upload OBJ File" and select your 3D molecular structure file
- The app supports standard OBJ format files

### 2. Set Camera Angles
- **Elevation** (-89° to 89°): Vertical viewing angle
  - Negative values: View from below
  - Positive values: View from above
- **Azimuth** (0° to 360°): Horizontal rotation
  - 0°: Front view
  - 90°: Right side view
  - 180°: Back view  
  - 270°: Left side view
- **Roll** (0° to 360°): Camera rotation around viewing axis

### 3. Configure AFM Parameters
- **Tip Radius** (0.1-10.0 nm): AFM tip size
  - Larger tips → Smoother, less detailed images
  - Smaller tips → More detailed, sharper features
- **Scan Step** (0.1-5.0 nm): Scanning resolution
  - Smaller steps → Higher resolution (slower)
  - Larger steps → Lower resolution (faster)

### 4. Generate AFM Image
- Click "Generate AFM Image" to process
- View results in the output panels:
  - **Virtual AFM Image**: Simulated AFM scan
  - **Raw Depth Map**: Unprocessed depth information

## Example Workflow

1. **Load a protein structure**: Upload your `.obj` file
2. **Set initial view**: Try elevation=45°, azimuth=0°, roll=0°
3. **Adjust AFM settings**: Start with tip radius=1.0nm, scan step=1.0nm
4. **Generate image**: Click the generate button
5. **Explore different angles**: Try different viewing angles to see various surface features
6. **Fine-tune parameters**: Adjust tip size and scan resolution as needed

## Advanced Options

- **Convert Angstrom to Nanometers**: Check this if your input coordinates are in Angstroms instead of nanometers
- **Render Size**: Internal rendering resolution (higher = better quality but slower)

## Tips for Best Results

- **File Format**: Ensure your OBJ file is properly formatted with vertex and face information
- **File Size**: Larger meshes may take longer to process
- **Parameter Balance**: Balance between detail (smaller tip/step) and processing speed
- **Viewing Angles**: Try multiple angles to capture different surface features
- **Progressive Adjustment**: Start with default parameters and adjust incrementally

## Troubleshooting

### Common Issues

1. **"Please upload an OBJ file" error**
   - Make sure you've selected a valid .obj file
   - Check that the file isn't corrupted

2. **Processing takes too long**
   - Reduce the render size
   - Increase the scan step size
   - Use a larger tip radius

3. **Low quality results**
   - Increase the render size
   - Decrease the scan step size
   - Use a smaller tip radius

4. **CUDA/GPU errors**
   - The app will automatically fall back to CPU if GPU isn't available
   - Make sure PyTorch is properly installed with CUDA support

### Performance Optimization

- **GPU Usage**: The app automatically uses GPU if available (CUDA)
- **Memory Management**: Large files are processed in chunks to manage memory
- **Caching**: Results are temporarily cached during processing

## Development Notes

- Built on top of the existing `py3d_virtual_afm.py` module
- Uses PyTorch3D for 3D rendering and mesh processing
- Gradio provides the web interface framework
- OpenCV handles image processing operations

## Requirements

- Python 3.7+
- PyTorch
- PyTorch3D
- Gradio 4.0+
- OpenCV
- NumPy
- Matplotlib

See `requirements.txt` for complete dependency list.
