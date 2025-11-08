import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from typing import Tuple, List, Optional

class QuadtreeNode:
    """Node in the quadtree structure."""
    def __init__(self, x: int, y: int, size: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.size = size
        self.color = color  # (R, G, B)
        self.is_leaf = True
        self.children = []  # [NW, NE, SW, SE]
        self.variance = 0.0
    
    def __repr__(self):
        return f"Node({self.x},{self.y},{self.size}) color={self.color} leaf={self.is_leaf}"

def compute_variance(region: np.ndarray) -> float:
    """
    Compute color variance in a region.
    Variance is the average squared distance from mean color.
    """
    if region.size == 0:
        return 0.0
    
    # Flatten spatial dimensions, keep color channels
    pixels = region.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) == 0:
        return 0.0
    
    mean_color = np.mean(pixels, axis=0)
    # Sum of squared distances from mean
    variance = np.sum((pixels - mean_color) ** 2) / len(pixels)
    
    return variance

def get_average_color(region: np.ndarray) -> Tuple[int, int, int]:
    """Compute average color of a region."""
    if region.size == 0:
        return (0, 0, 0)
    
    pixels = region.reshape(-1, 3)
    mean = np.mean(pixels, axis=0)
    return tuple(mean.astype(np.uint8))

def build_quadtree(image: np.ndarray, x: int, y: int, size: int, 
                   threshold: float) -> QuadtreeNode:
    """
    Build quadtree by recursively subdividing image regions.
    
    Args:
        image: RGB image array (height, width, 3)
        x, y: Top-left corner of current region
        size: Size of current region (square)
        threshold: Variance threshold for subdivision
    
    Returns:
        QuadtreeNode representing this region
    """
    # Extract region
    region = image[y:y+size, x:x+size]
    
    # Compute statistics
    avg_color = get_average_color(region)
    variance = compute_variance(region)
    
    # Create node
    node = QuadtreeNode(x, y, size, avg_color)
    node.variance = variance
    
    # Base case: homogeneous region or minimum size
    if variance <= threshold or size <= 1:
        node.is_leaf = True
        return node
    
    # Recursive case: subdivide into 4 quadrants
    half = size // 2
    if half == 0:
        node.is_leaf = True
        return node
    
    # Build children (NW, NE, SW, SE)
    nw = build_quadtree(image, x, y, half, threshold)
    ne = build_quadtree(image, x + half, y, half, threshold)
    sw = build_quadtree(image, x, y + half, half, threshold)
    se = build_quadtree(image, x + half, y + half, half, threshold)
    
    node.is_leaf = False
    node.children = [nw, ne, sw, se]
    
    return node

def decompress_image(quadtree: QuadtreeNode, width: int, height: int) -> np.ndarray:
    """
    Reconstruct image from quadtree.
    
    Args:
        quadtree: Root of quadtree
        width, height: Dimensions of output image
    
    Returns:
        Reconstructed RGB image
    """
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    def fill_region(node: QuadtreeNode):
        if node.is_leaf:
            # Fill entire region with node's color
            result[node.y:node.y+node.size, node.x:node.x+node.size] = node.color
        else:
            # Recursively fill children
            for child in node.children:
                fill_region(child)
    
    fill_region(quadtree)
    return result

def extract_palette(quadtree: QuadtreeNode) -> List[Tuple[int, int, int]]:
    """Extract unique colors (palette) from quadtree leaves."""
    palette = []
    
    def traverse(node: QuadtreeNode):
        if node.is_leaf:
            if node.color not in palette:
                palette.append(node.color)
        else:
            for child in node.children:
                traverse(child)
    
    traverse(quadtree)
    return palette

def count_leaves(quadtree: QuadtreeNode) -> int:
    """Count number of leaf nodes in quadtree."""
    if quadtree.is_leaf:
        return 1
    return sum(count_leaves(child) for child in quadtree.children)

def compute_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute Mean Squared Error between images."""
    diff = original.astype(np.float32) - compressed.astype(np.float32)
    mse = np.mean(diff ** 2)
    return mse

def compute_psnr(mse: float) -> float:
    """Compute Peak Signal-to-Noise Ratio from MSE."""
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ========== TEST IMAGE GENERATION ==========

def generate_test_image(size: int = 256) -> np.ndarray:
    """Generate a test image with varying detail levels."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background gradient (smooth region - should compress well)
    for i in range(size):
        for j in range(size):
            image[i, j] = [
                int(100 + 100 * i / size),  # Red gradient
                int(150),                     # Constant green
                int(200 - 100 * j / size)    # Blue gradient
            ]
    
    # Add detailed region (should create deep quadtree)
    center = size // 2
    detail_size = size // 4
    for i in range(detail_size):
        for j in range(detail_size):
            # Checkerboard pattern
            if (i // 4 + j // 4) % 2 == 0:
                color = [255, 0, 0]  # Red
            else:
                color = [0, 0, 255]  # Blue
            
            y = center - detail_size // 2 + i
            x = center - detail_size // 2 + j
            if 0 <= y < size and 0 <= x < size:
                image[y, x] = color
    
    # Add some circles (medium detail)
    for circle_idx in range(3):
        cx = size // 4 + circle_idx * size // 4
        cy = size // 4
        radius = size // 10
        
        for i in range(size):
            for j in range(size):
                if (i - cy) ** 2 + (j - cx) ** 2 < radius ** 2:
                    image[i, j] = [255, 255, 0]  # Yellow
    
    return image

# ========== EXPERIMENTS ==========

def visualize_compression(image: np.ndarray, thresholds: List[float]):
    """Visualize compression at different threshold levels."""
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, n_thresholds + 1, figsize=(4*(n_thresholds+1), 8))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    original_colors = len(np.unique(image.reshape(-1, 3), axis=0))
    
    for idx, threshold in enumerate(thresholds):
        # Compress
        start = time.time()
        quadtree = build_quadtree(image, 0, 0, image.shape[1], threshold)
        compress_time = time.time() - start
        
        # Decompress
        start = time.time()
        compressed = decompress_image(quadtree, image.shape[1], image.shape[0])
        decompress_time = time.time() - start
        
        # Metrics
        palette = extract_palette(quadtree)
        n_colors = len(palette)
        n_leaves = count_leaves(quadtree)
        mse = compute_mse(image, compressed)
        psnr = compute_psnr(mse)
        compression_ratio = original_colors / n_colors if n_colors > 0 else 0
        
        # Display compressed image
        axes[0, idx + 1].imshow(compressed)
        axes[0, idx + 1].set_title(
            f'τ = {threshold:.0f}\n{n_colors} colors',
            fontsize=11
        )
        axes[0, idx + 1].axis('off')
        
        # Display metrics
        metrics_text = (
            f'Leaves: {n_leaves}\n'
            f'Colors: {n_colors}\n'
            f'Compression: {compression_ratio:.1f}x\n'
            f'MSE: {mse:.2f}\n'
            f'PSNR: {psnr:.2f} dB\n'
            f'Compress: {compress_time*1000:.2f}ms\n'
            f'Decompress: {decompress_time*1000:.2f}ms'
        )
        axes[1, idx + 1].text(0.1, 0.5, metrics_text, fontsize=10, 
                              verticalalignment='center', family='monospace')
        axes[1, idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('compression_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: compression_comparison.png")
    plt.show()

def run_scaling_experiments():
    """Test algorithm scaling with image size."""
    sizes = [32, 64, 128, 256, 512]
    threshold = 500  # Fixed threshold
    
    compress_times = []
    decompress_times = []
    n_leaves_list = []
    
    print("\n" + "="*70)
    print("SCALING EXPERIMENTS")
    print("="*70)
    print(f"{'Size':<8} {'Pixels':<12} {'Compress(ms)':<15} {'Decompress(ms)':<17} {'Leaves':<10}")
    print("-"*70)
    
    for size in sizes:
        # Generate test image
        image = generate_test_image(size)
        n_pixels = size * size
        
        # Compression
        start = time.time()
        quadtree = build_quadtree(image, 0, 0, size, threshold)
        compress_time = (time.time() - start) * 1000  # Convert to ms
        
        # Decompression
        start = time.time()
        compressed = decompress_image(quadtree, size, size)
        decompress_time = (time.time() - start) * 1000  # Convert to ms
        
        n_leaves = count_leaves(quadtree)
        
        compress_times.append(compress_time)
        decompress_times.append(decompress_time)
        n_leaves_list.append(n_leaves)
        
        print(f"{size:<8} {n_pixels:<12} {compress_time:<15.2f} {decompress_time:<17.2f} {n_leaves:<10}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Compression time vs theoretical
    ax1 = axes[0]
    n_pixels = [s*s for s in sizes]
    
    ax1.plot(sizes, compress_times, 'bo-', label='Observed Compression Time', linewidth=2, markersize=8)
    
    # Theoretical O(n² log n)
    theoretical = [p * np.log2(s) for s, p in zip(sizes, n_pixels)]
    scale = compress_times[-1] / theoretical[-1]
    theoretical_scaled = [t * scale for t in theoretical]
    
    ax1.plot(sizes, theoretical_scaled, 'r--', label='O(n²logn)', linewidth=2)
    ax1.set_xlabel('Image Size (n×n)', fontsize=12)
    ax1.set_ylabel('Compression Time (ms)', fontsize=12)
    ax1.set_title('Compression Time Scaling', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of leaves
    ax2 = axes[1]
    ax2.plot(sizes, n_leaves_list, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Image Size (n×n)', fontsize=12)
    ax2.set_ylabel('Number of Leaves', fontsize=12)
    ax2.set_title('Quadtree Complexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaling_experiments.png', dpi=150, bbox_inches='tight')
    print("\nSaved: scaling_experiments.png")
    plt.show()

def run_threshold_experiments():
    """Test effect of threshold on compression quality."""
    image = generate_test_image(256)
    thresholds = np.logspace(1, 4, 20)  # 10 to 10000
    
    colors_list = []
    mse_list = []
    psnr_list = []
    
    print("\n" + "="*70)
    print("THRESHOLD EXPERIMENTS (Quality vs Compression Trade-off)")
    print("="*70)
    
    for threshold in thresholds:
        quadtree = build_quadtree(image, 0, 0, 256, threshold)
        compressed = decompress_image(quadtree, 256, 256)
        
        palette = extract_palette(quadtree)
        n_colors = len(palette)
        mse = compute_mse(image, compressed)
        psnr = compute_psnr(mse)
        
        colors_list.append(n_colors)
        mse_list.append(mse)
        psnr_list.append(psnr)
    
    # Plot trade-off
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.semilogx(thresholds, colors_list, 'b-', linewidth=2)
    ax1.set_xlabel('Threshold (τ)', fontsize=12)
    ax1.set_ylabel('Number of Colors', fontsize=12)
    ax1.set_title('Palette Size vs Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.semilogx(thresholds, psnr_list, 'r-', linewidth=2)
    ax2.set_xlabel('Threshold (τ)', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Quality vs Threshold', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=30, color='g', linestyle='--', label='Good Quality (30dB)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_experiments.png', dpi=150, bbox_inches='tight')
    print("Saved: threshold_experiments.png")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("IMAGE COMPRESSION: QUADTREE COLOR QUANTIZATION")
    print("="*70)
    
    # Generate test image
    print("\nGenerating test image...")
    test_image = generate_test_image(256)
    
    print(f"Image size: {test_image.shape[0]}×{test_image.shape[1]}")
    print(f"Original colors: {len(np.unique(test_image.reshape(-1, 3), axis=0))}")
    
    # Visualize compression at different thresholds
    print("\n" + "="*70)
    print("COMPRESSION VISUALIZATION")
    print("="*70)
    thresholds_to_test = [100, 500, 1000, 2000]
    visualize_compression(test_image, thresholds_to_test)
    
    # Run scaling experiments
    run_scaling_experiments()
    
    # Run threshold experiments
    run_threshold_experiments()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - compression_comparison.png")
    print("  - scaling_experiments.png")
    print("  - threshold_experiments.png")