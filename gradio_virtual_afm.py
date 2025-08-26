import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from py3d_virtual_afm import VirtualAFM
import cv2

class GradioVirtualAFM:
    def __init__(self):
        """Initialize the Gradio Virtual AFM interface."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.virtual_afm = VirtualAFM(self.device)
        
    def process_obj_file(self, obj_file, elevation, azimuth, roll, 
                        image_size=256, tip_radius_nm=1.0, scan_step_nm=1.0, 
                        angstrom_to_nm=False):
        """
        Process the uploaded OBJ file and generate Virtual AFM image.
        
        Args:
            obj_file: Uploaded OBJ file
            elevation: Elevation angle in degrees
            azimuth: Azimuthal angle in degrees  
            roll: Roll angle in degrees
            image_size: Size of the output image
            tip_radius_nm: AFM tip radius in nanometers
            scan_step_nm: Scanning step size in nanometers
            angstrom_to_nm: Whether to convert from Angstrom to nanometers
            
        Returns:
            tuple: (AFM image, raw depth map, info text)
        """
        try:
            if obj_file is None:
                return None, None, "Please upload an OBJ file."
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_file:
                # Handle both file objects and bytes
                if hasattr(obj_file, 'read'):
                    # If it's a file object
                    tmp_file.write(obj_file.read())
                else:
                    # If it's bytes or other data type
                    tmp_file.write(obj_file)
                tmp_path = tmp_file.name
                
            try:
                # Load the mesh
                self.virtual_afm.load_mesh(tmp_path, angstrom_to_nm=angstrom_to_nm)
                
                # Create rasterizer
                rasterizer = self.virtual_afm.create_rasterizer(image_size=image_size)
                
                # Generate camera with specified angles
                cameras, pose_params = self.virtual_afm.camera_pose_from_params(
                    elevation=elevation, 
                    azimuth=azimuth, 
                    roll=roll, 
                    distance=2.0
                )
                
                # Render the view
                depth_map = self.virtual_afm.render_view(cameras, rasterizer, in_nm=True)
                
                # Convert to numpy for processing
                depth_map_np = depth_map.cpu().numpy()
                
                # Calculate scan step size based on bounding box
                dsize = int(torch.ceil(self.virtual_afm.bbox_size / scan_step_nm).cpu().numpy())
                dsize = min(dsize, 512)  # Limit maximum size for performance
                
                # Resize depth map
                depth_map_resized = cv2.resize(depth_map_np, (dsize, dsize), 
                                             interpolation=cv2.INTER_NEAREST_EXACT)
                
                # Add padding for tip convolution
                pad_size = max(5, int(tip_radius_nm / scan_step_nm) + 2)
                depth_map_padded = np.pad(depth_map_resized, 
                                        ((pad_size, pad_size), (pad_size, pad_size)), 
                                        mode='constant', constant_values=0)
                
                # Perform tip convolution
                afm_image = self.virtual_afm.perform_tip_convolution(
                    depth_map_padded,
                    tip_radius_nm=tip_radius_nm,
                    nm_per_pixel=scan_step_nm,
                    background_value=0
                )
                
                # Resize to final display size
                afm_image_display = cv2.resize(afm_image, (256, 256))
                depth_map_display = cv2.resize(depth_map_np, (256, 256))
                
                # Normalize images for Gradio display (convert to 0-1 range)
                def normalize_image_for_display(img):
                    """Normalize image values to 0-1 range for Gradio display."""
                    if img is None or img.size == 0:
                        return None
                    
                    # Remove any infinite or NaN values
                    img_clean = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Find valid range (exclude background/zero values)
                    valid_mask = img_clean > 0
                    if np.any(valid_mask):
                        valid_values = img_clean[valid_mask]
                        min_val = np.min(valid_values)
                        max_val = np.max(valid_values)
                        
                        if max_val > min_val:
                            # Normalize to 0-1 range
                            img_normalized = np.zeros_like(img_clean, dtype=np.float32)
                            img_normalized[valid_mask] = (img_clean[valid_mask] - min_val) / (max_val - min_val)
                        else:
                            # All values are the same
                            img_normalized = np.where(valid_mask, 1.0, 0.0)
                    else:
                        # No valid values, return zeros
                        img_normalized = np.zeros_like(img_clean, dtype=np.float32)
                    
                    return img_normalized
                
                # Normalize both images for display
                afm_image_display = normalize_image_for_display(afm_image_display)
                depth_map_display = normalize_image_for_display(depth_map_display)
                
                # Create info text
                info = f"""
                Mesh Information:
                - Bounding box size: {self.virtual_afm.bbox_size:.2f} nm
                - Camera pose: Elevation={elevation}°, Azimuth={azimuth}°, Roll={roll}°
                - AFM parameters: Tip radius={tip_radius_nm} nm, Scan step={scan_step_nm} nm
                - Image size: {dsize}x{dsize} pixels (rescaled to 256x256 for display)
                - Device: {self.device}
                """
                
                # Add image statistics to info
                if afm_image is not None:
                    afm_stats = f"""
                Image Statistics:
                - AFM Image range: {np.min(afm_image):.2f} to {np.max(afm_image):.2f} nm
                - Depth Map range: {np.min(depth_map_np):.2f} to {np.max(depth_map_np):.2f} nm
                - Normalized for display: 0.0 to 1.0
                """
                    info += afm_stats
                
                return afm_image_display, depth_map_display, info
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            error_msg = f"Error processing OBJ file: {str(e)}"
            print(error_msg)
            return None, None, error_msg

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the Virtual AFM processor
    afm_processor = GradioVirtualAFM()
    
    # Create the Gradio interface
    with gr.Blocks(title="Virtual AFM - Atomic Force Microscopy Simulation", 
                   theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔬 Virtual AFM - Atomic Force Microscopy Simulation
        
        Upload a 3D molecular structure (OBJ file) and simulate AFM imaging with customizable parameters.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Parameters")
                
                # File upload
                obj_file = gr.File(
                    label="Upload OBJ File",
                    file_types=[".obj"],
                    type="binary"
                )
                
                # Camera angles
                with gr.Group():
                    gr.Markdown("**Camera Orientation**")
                    elevation = gr.Slider(
                        minimum=-89, maximum=89, value=45, step=1,
                        label="Elevation Angle (degrees)",
                        info="Vertical viewing angle (-89 to 89)"
                    )
                    azimuth = gr.Slider(
                        minimum=0, maximum=360, value=0, step=1,
                        label="Azimuthal Angle (degrees)", 
                        info="Horizontal rotation (0 to 360)"
                    )
                    roll = gr.Slider(
                        minimum=0, maximum=360, value=0, step=1,
                        label="Roll Angle (degrees)",
                        info="Camera roll rotation (0 to 360)"
                    )
                
                # AFM parameters
                with gr.Group():
                    gr.Markdown("**AFM Parameters**")
                    tip_radius = gr.Slider(
                        minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                        label="Tip Radius (nm)",
                        info="AFM tip radius in nanometers"
                    )
                    scan_step = gr.Slider(
                        minimum=0.1, maximum=5.0, value=1.0, step=0.1,
                        label="Scan Step (nm)",
                        info="Scanning step size in nanometers"
                    )
                    image_size = gr.Slider(
                        minimum=64, maximum=512, value=256, step=64,
                        label="Render Size",
                        info="Internal rendering resolution"
                    )
                
                # Advanced options
                with gr.Accordion("Advanced Options", open=False):
                    angstrom_to_nm = gr.Checkbox(
                        label="Convert Angstrom to Nanometers",
                        value=False,
                        info="Check if input coordinates are in Angstroms"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "Generate AFM Image", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                
                with gr.Row():
                    afm_output = gr.Image(
                        label="Virtual AFM Image",
                        type="numpy",
                        height=300
                    )
                    depth_output = gr.Image(
                        label="Raw Depth Map", 
                        type="numpy",
                        height=300
                    )
                
                info_output = gr.Textbox(
                    label="Processing Information",
                    lines=8,
                    max_lines=12
                )
        
        # Examples section
        gr.Markdown("### 💡 Tips")
        gr.Markdown("""
        - **Elevation**: Negative values look from below, positive from above
        - **Azimuth**: 0° = front view, 90° = right side, 180° = back, 270° = left side  
        - **Roll**: Rotates the camera around its viewing axis
        - **Tip Radius**: Larger tips create smoother, less detailed images
        - **Scan Step**: Smaller steps give higher resolution but take more processing time
        """)
        
        # Connect the interface
        generate_btn.click(
            fn=afm_processor.process_obj_file,
            inputs=[
                obj_file, elevation, azimuth, roll, 
                image_size, tip_radius, scan_step, angstrom_to_nm
            ],
            outputs=[afm_output, depth_output, info_output]
        )
        
        # Auto-generate on parameter change (optional - can be enabled)
        # for input_component in [elevation, azimuth, roll, tip_radius, scan_step]:
        #     input_component.change(
        #         fn=afm_processor.process_obj_file,
        #         inputs=[obj_file, elevation, azimuth, roll, image_size, tip_radius, scan_step, angstrom_to_nm],
        #         outputs=[afm_output, depth_output, info_output]
        #     )
    
    return demo

def main():
    """Main function to launch the Gradio app."""
    try:
        # Create the interface
        demo = create_gradio_interface()
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create public link
            debug=True,             # Enable debug mode
            show_error=True         # Show detailed error messages
        )
        
    except KeyboardInterrupt:
        print("\nShutting down Virtual AFM Gradio app...")
    except Exception as e:
        print(f"Error launching app: {str(e)}")

if __name__ == "__main__":
    main()
