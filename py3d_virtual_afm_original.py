import os
import sys
import torch
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRasterizer, RasterizationSettings,
    FoVOrthographicCameras, look_at_view_transform
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_euler_angles, euler_angles_to_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import argparse

class MeshScaler:
    """Handle scaling between physical (nm) and normalized coordinates."""
    def __init__(self, verts_nm):
        """
        Initialize the MeshScaler with vertex coordinates in nanometers.

        Args:
            verts_nm (torch.Tensor): Vertex coordinates in nanometers.
        """
        self.min_coords = verts_nm.min(dim=0).values
        self.max_coords = verts_nm.max(dim=0).values
        self.center = (self.min_coords + self.max_coords) / 2.0
        self.bbox_size = (self.max_coords - self.min_coords).max()
        self.scale = 2.0 / self.bbox_size

    def normalize_mesh(self, verts_nm):
        """
        Convert vertex coordinates from nanometers to normalized coordinates [-1, 1].

        Args:
            verts_nm (torch.Tensor): Vertex coordinates in nanometers.

        Returns:
            torch.Tensor: Normalized vertex coordinates.
        """
        return (verts_nm - self.center) * self.scale

class VirtualAFM:
    def __init__(self, device):
        """
        Initialize the VirtualAFM with the specified device.

        Args:
            device (torch.device): The device to use for computations.
        """
        self.device = device

    def load_mesh(self, mesh_path):
        """
        Load a mesh from the specified path and normalize its vertex coordinates.

        Args:
            mesh_path (str): Path to the mesh file.
        """
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        verts = mesh.verts_packed()
        verts_nm = verts * 0.1  # Convert verts from Angstrom to nanometers
        mesh_scaler = MeshScaler(verts_nm)
        verts_normalized = mesh_scaler.normalize_mesh(verts_nm)
        self.scale = mesh_scaler.scale
        self.center = mesh_scaler.center
        self.bbox_size = mesh_scaler.bbox_size
        self.mesh_normalized = Meshes(verts=[verts_normalized], faces=mesh.faces_list())

    def create_rasterizer(self, image_size=100, blur_radius=0.0, faces_per_pixel=1, bin_size=-1):
        """
        Create a MeshRasterizer with the specified settings.

        Args:
            image_size (int): Size of the output image.
            blur_radius (float): Radius for blurring.
            faces_per_pixel (int): Number of faces per pixel.
            bin_size (int): Bin size for rasterization.

        Returns:
            MeshRasterizer: The created rasterizer.
        """
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size
        )
        return MeshRasterizer(raster_settings=raster_settings)

    def invert_depth_map(self, depth_map, background_value=-1):
        """
        Invert the depth map.

        Args:
            depth_map (torch.Tensor): The depth map to invert.
            background_value (float): The value representing the background.

        Returns:
            torch.Tensor: The inverted depth map.
        """
        mask = depth_map != background_value
        valid_depth = depth_map[mask]
        if valid_depth.numel() == 0:
            return torch.zeros_like(depth_map)
        min_depth, max_depth = valid_depth.min(), valid_depth.max()
        depth_map[mask] = max_depth - (depth_map[mask] - min_depth)
        depth_map[~mask] = 0.0
        return depth_map

    def random_camera_pose(self, elev_range=(-90, 90), azim_range=(0, 360), roll_range=(0, 360), dist=2.0):
        """
        Generate a random camera pose.

        Args:
            elev_range (tuple): Range of elevation angles.
            azim_range (tuple): Range of azimuth angles.
            roll_range (tuple): Range of roll angles.
            dist (float): Distance from the camera to the object.

        Returns:
            FoVOrthographicCameras: The camera object.
            dict: The pose parameters.
        """
        elev = np.random.randint(*elev_range)
        azim = np.random.randint(*azim_range)
        roll = np.random.randint(*roll_range)
        if elev == 90:
            elev = 89
        elif elev == -90:
            elev = -89
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        roll_rad = torch.tensor([[0, 0, roll]]) * (torch.pi / 180.0)
        R_roll = euler_angles_to_matrix(roll_rad, "XYZ")
        R_final = torch.bmm(R, R_roll)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R_final,
            T=T,
            min_x=-1.1,
            max_x=1.1,
            min_y=-1.1,
            max_y=1.1,
            znear=-1.1,
            zfar=1.1
        )
        pose_params = {'elevation': elev, 'azimuth': azim, 'roll': roll, 'distance': dist}
        return cameras, pose_params

    def camera_pose_from_params(self, elevation=0.0, azimuth=0.0, roll=0.0, distance=2.0):
        """
        Generate a camera pose from the specified parameters.

        Args:
            elevation (float): Elevation angle.
            azimuth (float): Azimuth angle.
            roll (float): Roll angle.
            distance (float): Distance from the camera to the object.

        Returns:
            FoVOrthographicCameras: The camera object.
            dict: The pose parameters.
        """
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        roll_rad = torch.tensor([[0, 0, roll]]) * (torch.pi / 180.0)
        R_roll = euler_angles_to_matrix(roll_rad, "XYZ")
        R_final = torch.bmm(R, R_roll)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R_final,
            T=T,
            min_x=-1.1,
            max_x=1.1,
            min_y=-1.1,
            max_y=1.1,
            znear=-1.1,
            zfar=1.1
        )
        pose_params = {'elevation': elevation, 'azimuth': azimuth, 'roll': roll, 'distance': distance}
        return cameras, pose_params

    def convert_height_map_to_nm(self, height_map, background_value=-1):
        """
        Convert the height map from normalized coordinates to nanometers.

        Args:
            height_map (torch.Tensor): The height map to convert.
            background_value (float): The value representing the background.

        Returns:
            torch.Tensor: The height map in nanometers.
        """
        height_map_nm = torch.where(height_map != background_value, ((height_map - 1.0) / self.scale) + self.center[2], background_value)
        return height_map_nm

    def render_view(self, cameras, rasterizer, in_nm=False, background_value=-1):
        """
        Render a view of the mesh.

        Args:
            cameras (FoVOrthographicCameras): The camera object.
            rasterizer (MeshRasterizer): The rasterizer object.
            in_nm (bool): Whether to convert the height map to nanometers.
            background_value (float): The value representing the background.

        Returns:
            torch.Tensor: The rendered height map.
        """
        fragments = rasterizer(self.mesh_normalized, cameras=cameras)
        depth_map = fragments.zbuf[0, ..., 0]
        height_map = self.invert_depth_map(depth_map, background_value=background_value)
        if in_nm:
            height_map = self.convert_height_map_to_nm(height_map, background_value=0.0)
        return height_map

    def render_multiple_views(self, num_views=20, image_size=100, in_nm=False, background_value=-1):
        """
        Render multiple views of the mesh.

        Args:
            num_views (int): Number of views to render.
            image_size (int): Size of the output images.
            in_nm (bool): Whether to convert the height maps to nanometers.
            background_value (float): The value representing the background.

        Returns:
            list: A list of tuples containing the rendered height maps and pose parameters.
        """
        rasterizer = self.create_rasterizer(image_size)
        rendered_views = []
        for _ in range(num_views):
            cameras, pose_params = self.random_camera_pose()
            depth_map = self.render_view(cameras, rasterizer, in_nm=in_nm, background_value=background_value)
            rendered_views.append((depth_map, pose_params))
        return rendered_views

    def spherical_selem(self, radius_nm, nm_per_pixel):
        """
        Create a spherical structuring element.

        Args:
            radius_nm (float): Radius of the sphere in nanometers.
            nm_per_pixel (float): Nanometers per pixel.

        Returns:
            np.ndarray: The spherical structuring element.
        """
        r_px = radius_nm / nm_per_pixel
        size = int(np.ceil(2 * r_px)) + 1
        selem = -np.inf * np.ones((size, size), dtype=float)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dx = i - center
                dy = j - center
                dist_sq = dx * dx + dy * dy
                if dist_sq <= r_px * r_px:
                    dist = np.sqrt(dist_sq)
                    z = np.sqrt(r_px * r_px - dist_sq)
                    selem[i, j] = z
        selem = selem * nm_per_pixel
        return selem

    def nonflat_grayscale_dilation(self, img, selem):
        """
        Perform non-flat grayscale dilation on an image.

        Args:
            img (np.ndarray): The input image.
            selem (np.ndarray): The structuring element.

        Returns:
            np.ndarray: The dilated image.
        """
        H, W = img.shape
        M, N = selem.shape
        mC, nC = M // 2, N // 2
        padded_img = np.pad(img, ((mC, M - mC - 1), (nC, N - nC - 1)), mode='constant', constant_values=-np.inf)
        dilated = np.full_like(img, -np.inf, dtype=float)
        for i in range(M):
            for j in range(N):
                if selem[i, j] == -np.inf:
                    continue
                shifted_img = padded_img[i:i + H, j:j + W] + selem[i, j]
                dilated = np.maximum(dilated, shifted_img)
        return dilated

    def nonflat_grayscale_dilation2(self, img, selem, background_value=0):
        """
        Perform non-flat grayscale dilation on an image with a specified background value.

        Args:
            img (np.ndarray): The input image.
            selem (np.ndarray): The structuring element.
            background_value (float): The value representing the background.

        Returns:
            np.ndarray: The dilated image.
        """
        H, W = img.shape
        M, N = selem.shape
        mC, nC = M // 2, N // 2
        padded_img = np.pad(img, ((mC, M - mC - 1), (nC, N - nC - 1)), mode='constant', constant_values=background_value)
        dilated = np.full_like(img, -np.inf, dtype=float)
        has_effect = np.zeros_like(img, dtype=bool)
        valid_selem = selem != -np.inf
        for i in range(M):
            for j in range(N):
                if not valid_selem[i, j]:
                    continue
                window = padded_img[i:i + H, j:j + W]
                candidate = window + selem[i, j]
                dilated = np.maximum(dilated, candidate)
                has_effect |= (window != background_value)
        result = np.where(has_effect, dilated, background_value)
        return result

    def differentiable_grayscale_dilation(self, img, selem, background_value=0):
        """
        Differentiable version of grayscale dilation using PyTorch operations.
        
        Args:
            img (torch.Tensor): Input image [H, W]
            selem (torch.Tensor): Structuring element [M, N]
            background_value (float): Background value
        """
        H, W = img.shape
        M, N = selem.shape
        mC, nC = M // 2, N // 2
        
        # Pad the image using PyTorch
        padded_img = F.pad(img.unsqueeze(0).unsqueeze(0), 
                        (nC, N - nC - 1, mC, M - mC - 1),
                        mode='constant', 
                        value=background_value)
        # Use unfold to create sliding windows
        windows = F.unfold(padded_img, 
                        kernel_size=(M, N),
                        stride=1).reshape(M*N, H, W)
        # Add the structuring element values
        windows = windows + selem.reshape(-1, 1, 1)
        # Max over the kernel dimension
        dilated = torch.max(windows, dim=0)[0]
        # make the background value to be 0
        dilated[dilated <= selem.max()] = 0
        return dilated

    def perform_tip_convolution(self, img, selem=None, tip_radius_nm=1.0, nm_per_pixel=1.0, background_value=0):
        """
        Perform tip convolution on an image.

        Args:
            img (np.ndarray): The input image.
            selem (np.ndarray): The structuring element.
            background_value (float): The value representing the background.

        Returns:
            np.ndarray: The convolved image.
        """
        if selem is None:
            selem = self.spherical_selem(tip_radius_nm, nm_per_pixel)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).cuda()
            img.requires_grad = True
        if isinstance(selem, np.ndarray):
            selem = torch.from_numpy(selem).cuda()
            selem.requires_grad = True
        # dilated = self.nonflat_grayscale_dilation2(img, selem, background_value)
        dilated = self.differentiable_grayscale_dilation(img, selem, background_value)
        dilated = dilated.detach().cpu().numpy()
        return dilated


    def plot_depth_maps(self, rendered_views, save_path=None, figsize=(20, 16)):
        """
        Plot the rendered depth maps.

        Args:
            rendered_views (list): A list of tuples containing the rendered depth maps and pose parameters.
            save_path (str): Path to save the plot.
            figsize (tuple): Size of the figure.

        Returns:
            None
        """
        num_views = len(rendered_views)
        rows = int(np.ceil(num_views / 5))
        cols = min(5, num_views)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axs = [axs]
        axs = np.array(axs).ravel()
        for idx, (depth_map, pose_params) in enumerate(rendered_views):
            depth_np = depth_map.cpu().numpy()
            im = axs[idx].imshow(depth_np, cmap='afmhot', vmin=0.0)
            title = f'e:{pose_params["elevation"]:.1f}° \n' \
                    f'a:{pose_params["azimuth"]:.1f}° \n' \
                    f'r:{pose_params["roll"]:.1f}°'
            axs[idx].set_title(title)
            plt.colorbar(im, ax=axs[idx], label='Depth (nm)')
        for idx in range(len(rendered_views), len(axs)):
            axs[idx].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def process_single_view(virtual_afm, view_idx, rasterizer, save_dir, tip_radius_nm=1.0, scan_step_nm=1.0):
    """Process and save a single view of the protein surface.
    
    Args:
        virtual_afm (VirtualAFM): The VirtualAFM instance
        view_idx (int): Index of the current view
        save_dir (str): Directory to save the outputs
        max_image_size (int): Maximum size of the rendered image
        tip_radius_nm (float): Radius of the AFM tip in nanometers
        scan_step_nm (float): Step size for scanning in nanometers
    """
    # Get camera view
    cameras, pose_params = virtual_afm.random_camera_pose()
    
    # Generate initial depth map
    depth_map = virtual_afm.render_view(cameras, rasterizer, in_nm=True)
    
    # Process depth map
    dsize = int(torch.ceil(virtual_afm.bbox_size / scan_step_nm).cpu().numpy())
    depth_map_array = depth_map.clone().cpu().numpy()
    depth_map_downsampled = cv2.resize(depth_map_array, (dsize, dsize), 
                                     interpolation=cv2.INTER_NEAREST_EXACT)
    
    # Add padding and perform tip convolution
    pad_size = 5
    depth_map_padded = np.pad(depth_map_downsampled, ((pad_size, pad_size), (pad_size, pad_size)), 
                             mode='constant', constant_values=0)
    depth_map_tip_convolved = virtual_afm.perform_tip_convolution(
        depth_map_padded, 
        tip_radius_nm=tip_radius_nm, 
        nm_per_pixel=scan_step_nm, 
        background_value=0
    )
    
    # Generate masks and final resized maps
    mask = (depth_map_tip_convolved != 0).astype(np.float32)
    mask_upsampled = (cv2.resize(mask, (256, 256)) > 0.5).astype(np.uint8)
    depth_map_256 = cv2.resize(depth_map_tip_convolved, (256, 256))
    
    # Save results
    np.savez_compressed(
        f"{save_dir}/depth_map_{view_idx}.npz",
        bbox_size=float(virtual_afm.bbox_size.cpu().numpy()), # bbox size in nm
        derived_afm_image_size=dsize, # derived afm image size in pixels
        depth_map_og=depth_map.cpu().numpy(), # original depth map in nm
        depth_map_without_tip_convolution=depth_map_padded, # depth map without tip convolution in nm
        depth_map_with_tip_convolution=depth_map_tip_convolved, # depth map with tip convolution in nm
        depth_map_with_tip_convolution_256=depth_map_256, # depth map with tip convolution in 256x256 pixels in nm
        mask_with_tip_convolution=mask, # mask with tip convolution
        mask_upsampled_256=mask_upsampled,
        pose_params=pose_params
    )
    
    # Save visualization
    plt.imsave(f"{save_dir}/depth_map_with_tip_convolution_256_{view_idx}.png", 
               depth_map_256, cmap='afmhot')

def main(args):
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize VirtualAFM
    protein_name = args.protein_name
    if protein_name is None:
        print("Please provide a protein name")
        return
    else:
        input_dir = args.input_dir
        try:
            mesh_path = f"{input_dir}/{protein_name}_surface_blob.obj"
            save_dir = f"{args.save_dir}/{protein_name}"
        except:
            print(f"Error: {protein_name}_surface_blob.obj not found")
            return
    
    # Create save directory and load mesh
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize VirtualAFM and create rasterizer
    virtual_afm = VirtualAFM(device)
    virtual_afm.load_mesh(mesh_path)
    max_image_size = args.max_image_size
    rasterizer = virtual_afm.create_rasterizer(max_image_size)
    
    # Process multiple views
    num_views = args.num_views
    for view_idx in range(num_views):
        process_single_view(virtual_afm, view_idx, rasterizer, save_dir, tip_radius_nm=args.tip_radius_nm, scan_step_nm=args.scan_step_nm)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default="cuda:0")
    args.add_argument("--input_dir", type=str, default="./example")
    args.add_argument("--save_dir", type=str, default="./example")
    args.add_argument("--protein_name", type=str, default=None)
    args.add_argument("--num_views", type=int, default=25)
    args.add_argument("--max_image_size", type=int, default=512)
    args.add_argument("--tip_radius_nm", type=float, default=1.0)
    args.add_argument("--scan_step_nm", type=float, default=1.0)
    args = args.parse_args()

    main(args)