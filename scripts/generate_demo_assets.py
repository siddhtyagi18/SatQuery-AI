import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "demo")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1024, 768

def create_base_landscape():
    """Generates an authentic looking satellite optical scene."""
    # Base terrain gradient (e.g. mix of agricultural greens, dry land, soil)
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Generate perlin-like low frequency noise for natural terrain variation
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Base soil/earth tone
    r = 30 + 15 * np.sin(x_coords / 120.0) + 10 * np.cos(y_coords / 90.0)
    g = 55 + 25 * np.cos(x_coords / 140.0) + 15 * np.sin(y_coords / 100.0)
    b = 35 + 10 * np.sin((x_coords + y_coords) / 110.0)
    
    arr[:, :, 0] = np.clip(r, 20, 80)
    arr[:, :, 1] = np.clip(g, 40, 110)
    arr[:, :, 2] = np.clip(b, 25, 60)
    
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    
    # Agricultural field parcels (irregular quadrilaterals)
    field_colors = [
        (34, 78, 45),    # dark green vegetation
        (48, 102, 54),   # vibrant green crop
        (65, 118, 60),   # light green field
        (85, 115, 65),   # olive green
        (105, 110, 60),  # yellowish crop
        (75, 70, 50),    # tilled fallow soil
        (90, 85, 62),    # dry agricultural parcel
        (55, 88, 52),    # mature crop
    ]
    
    # Draw patchwork fields on western and central areas
    for col in range(12):
        for row in range(9):
            x1 = int(col * 85 + random.randint(-15, 15))
            y1 = int(row * 85 + random.randint(-15, 15))
            x2 = int(x1 + 80 + random.randint(-10, 20))
            y2 = int(y1 + 80 + random.randint(-10, 20))
            
            poly = [
                (x1, y1),
                (x2, y1 + random.randint(-8, 8)),
                (x2 + random.randint(-10, 10), y2),
                (x1 + random.randint(-8, 8), y2)
            ]
            c = random.choice(field_colors)
            # Add field texture
            draw.polygon(poly, fill=c, outline=(25, 45, 30))
            
            # Subtle crop rows inside fields
            if random.random() > 0.5:
                angle_step = random.choice([4, 6, 8])
                for line_y in range(y1, y2, angle_step):
                    draw.line([(x1, line_y), (x2, line_y)], fill=(c[0]-8, c[1]-8, c[2]-8), width=1)
                    
    # Meandering river / water body in the south-west
    river_points = []
    curr_x, curr_y = 0, 520
    while curr_x < 650:
        river_points.append((curr_x, curr_y))
        curr_x += random.randint(25, 45)
        curr_y += random.randint(-15, 20)
    river_points.append((650, curr_y))
    
    # Draw water river with banks
    for i in range(len(river_points) - 1):
        draw.line([river_points[i], river_points[i+1]], fill=(35, 60, 50), width=28)
        draw.line([river_points[i], river_points[i+1]], fill=(14, 38, 55), width=20)
        draw.line([river_points[i], river_points[i+1]], fill=(20, 55, 75), width=12)
        
    # Rural roads / paths
    roads = [
        [(50, 120), (280, 160), (480, 240), (520, 450), (490, 750)],
        [(480, 240), (750, 210), (1000, 230)],
        [(280, 160), (310, 500), (330, 720)]
    ]
    for r in roads:
        draw.line(r, fill=(110, 105, 95), width=4)
        draw.line(r, fill=(140, 135, 125), width=2)
        
    # Small existing village clusters in west
    for (vx, vy) in [(180, 210), (240, 360), (380, 170), (120, 480)]:
        for _ in range(8):
            bx = vx + random.randint(-40, 40)
            by = vy + random.randint(-40, 40)
            bw, bh = random.randint(10, 18), random.randint(10, 18)
            draw.rectangle([bx, by, bx + bw, by + bh], fill=(130, 110, 95), outline=(60, 50, 45))
            
    return img

def generate_before_image():
    """Generates Cartosat-3 style optical baseline (2022)."""
    img = create_base_landscape()
    draw = ImageDraw.Draw(img)
    
    # In 2022 (Before): The North-Eastern sector (x: 550..1000, y: 80..520) is pure lush farmland & wetlands
    ne_greens = [(38, 92, 48), (45, 110, 55), (30, 80, 40), (52, 118, 62), (60, 125, 68)]
    for col in range(7):
        for row in range(6):
            x1 = 540 + col * 68 + random.randint(-10, 10)
            y1 = 70 + row * 75 + random.randint(-10, 10)
            x2 = x1 + 65 + random.randint(-8, 12)
            y2 = y1 + 72 + random.randint(-8, 12)
            c = random.choice(ne_greens)
            draw.polygon([(x1, y1), (x2, y1+4), (x2-2, y2), (x1+3, y2)], fill=c, outline=(20, 50, 28))
            # Crop rows
            for ly in range(y1, y2, 6):
                draw.line([(x1, ly), (x2, ly)], fill=(c[0]+8, c[1]+10, c[2]+8), width=1)
                
    # Add subtle sensor noise and blur for natural satellite look
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img

def generate_after_image():
    """Generates Cartosat-3 style optical scene in 2024 with urban expansion."""
    # Start from the exact same landscape base
    random.seed(42)
    np.random.seed(42)
    img = create_base_landscape()
    draw = ImageDraw.Draw(img)
    
    # In 2024 (After):
    # 1. New Major Road / Highway bisecting the scene
    highway = [(40, 310), (320, 305), (580, 290), (820, 270), (1010, 260)]
    draw.line(highway, fill=(50, 52, 58), width=14)
    draw.line(highway, fill=(75, 78, 85), width=10)
    draw.line(highway, fill=(210, 210, 210), width=1) # Center line
    
    # 2. Heavy Urban Expansion in North-East sector (x: 540..980, y: 80..520)
    # Cleared ground / graded earth foundation
    draw.rectangle([540, 75, 980, 520], fill=(85, 82, 78), outline=(65, 62, 58))
    
    # Commercial & residential building clusters
    building_roofs = [
        (185, 188, 192), # light concrete
        (215, 218, 222), # bright white reflective roof
        (140, 142, 148), # medium gray slab
        (165, 115, 90),  # terracotta tile
        (95, 115, 135),  # blue industrial warehouse
        (110, 112, 118), # asphalt / dark gray
    ]
    
    # Grid street layout in new development
    for sy in range(95, 510, 60):
        draw.line([(550, sy), (970, sy)], fill=(55, 56, 60), width=6)
    for sx in range(560, 970, 70):
        draw.line([(sx, 85), (sx, 515)], fill=(55, 56, 60), width=6)
        
    # Buildings
    random.seed(101)
    for bx_start in range(570, 950, 48):
        for by_start in range(105, 500, 42):
            if random.random() > 0.15:
                bw = random.randint(24, 38)
                bh = random.randint(20, 32)
                roof = random.choice(building_roofs)
                # Building shadow
                draw.rectangle([bx_start+3, by_start+3, bx_start+bw+3, by_start+bh+3], fill=(35, 36, 40))
                # Building roof
                draw.rectangle([bx_start, by_start, bx_start+bw, by_start+bh], fill=roof, outline=(40, 42, 46))
                
    # Add subtle sensor blur
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img

def generate_change_mask():
    """Generates the transparent RGBA change mask overlay matching the legend."""
    # RGBA image: 100% transparent everywhere
    mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    
    # 1. New Urban / Built-up (#FF5C5C with ~85% opacity = 216 alpha)
    urban_color = (255, 92, 92, 210)
    # Highlight the new building zone
    draw.rounded_rectangle([550, 85, 970, 515], radius=8, fill=(255, 92, 92, 175), outline=(255, 92, 92, 255), width=2)
    
    # 2. Vegetation Loss (#FFB020 with ~75% opacity = 190 alpha)
    # Surrounding buffer cleared for infrastructure & construction
    veg_loss_color = (255, 176, 32, 190)
    draw.polygon([(460, 80), (545, 80), (545, 530), (460, 530)], fill=veg_loss_color)
    draw.polygon([(545, 520), (980, 520), (980, 560), (545, 560)], fill=veg_loss_color)
    
    # 3. New Infrastructure (#3ED0FF with ~90% opacity = 230 alpha)
    # The new highway corridor
    highway_mask_color = (62, 208, 255, 235)
    highway = [(40, 310), (320, 305), (580, 290), (820, 270), (1010, 260)]
    draw.line(highway, fill=highway_mask_color, width=16)
    
    # Smooth edges slightly
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    return mask

def generate_optical_sample():
    """Generates crisp optical earth observation scene."""
    img = create_base_landscape()
    # Add coastline or reservoir in top right
    draw = ImageDraw.Draw(img)
    draw.polygon([(700, 0), (1024, 0), (1024, 400), (820, 280), (740, 120)], fill=(18, 48, 72))
    # Harbor / industrial port pier
    draw.polygon([(820, 280), (870, 240), (890, 265), (840, 305)], fill=(120, 125, 130))
    # Ships / vessels
    draw.rectangle([900, 180, 930, 192], fill=(230, 230, 235), outline=(40, 40, 50))
    draw.rectangle([940, 260, 965, 270], fill=(220, 100, 90), outline=(40, 40, 50))
    return img.filter(ImageFilter.GaussianBlur(0.5))

def generate_sar_sample():
    """Generates synthetic aperture radar (SAR) C-Band VV imagery."""
    # SAR has characteristic speckle and grayscale intensity backscatter
    sar_arr = np.random.gamma(shape=2.5, scale=22.0, size=(H, W)).astype(np.float32)
    
    # Water has very low backscatter (dark/near-zero specular reflection)
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    # Water mask in top right
    water_mask = ((x_coords > 700) & (y_coords < 300) & ((x_coords - 700) * 0.8 > y_coords - 50))
    sar_arr[water_mask] *= 0.15
    
    # Urban areas have extremely bright double-bounce reflections
    urban_mask = ((x_coords > 550) & (x_coords < 950) & (y_coords > 90) & (y_coords < 500))
    sar_arr[urban_mask] *= 2.2
    
    # Agricultural fields have medium textured backscatter
    sar_arr = np.clip(sar_arr, 0, 255).astype(np.uint8)
    
    # Convert to grayscale 3-channel RGB image
    sar_img = Image.fromarray(np.stack([sar_arr, sar_arr, sar_arr], axis=-1), mode="RGB")
    return sar_img

def main():
    print("Generating authentic demo satellite imagery binary assets...")
    
    # 1. Optical Before
    before = generate_before_image()
    before_path = os.path.join(OUT_DIR, "optical_before.jpg")
    before.save(before_path, "JPEG", quality=92)
    print(f"Saved {before_path} ({os.path.getsize(before_path)} bytes)")
    
    # 2. Optical After
    after = generate_after_image()
    after_path = os.path.join(OUT_DIR, "optical_after.jpg")
    after.save(after_path, "JPEG", quality=92)
    print(f"Saved {after_path} ({os.path.getsize(after_path)} bytes)")
    
    # 3. Change Mask PNG (RGBA)
    mask = generate_change_mask()
    mask_path = os.path.join(OUT_DIR, "change_mask.png")
    mask.save(mask_path, "PNG")
    print(f"Saved {mask_path} ({os.path.getsize(mask_path)} bytes)")
    
    # 4. Optical Sample
    optical = generate_optical_sample()
    optical_path = os.path.join(OUT_DIR, "optical_sample.jpg")
    optical.save(optical_path, "JPEG", quality=92)
    print(f"Saved {optical_path} ({os.path.getsize(optical_path)} bytes)")
    
    # 5. SAR Sample
    sar = generate_sar_sample()
    sar_path = os.path.join(OUT_DIR, "sar_sample.jpg")
    sar.save(sar_path, "JPEG", quality=92)
    print(f"Saved {sar_path} ({os.path.getsize(sar_path)} bytes)")
    
    print("All assets successfully generated!")

if __name__ == "__main__":
    main()
