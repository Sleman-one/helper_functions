import os
import glob

def make_dataset(path_to_image_folder = str,
                 path_to_mask_folder = str,
                 labe_string = "_L"
                 ):
    list_of_image_paths = sorted(glob.glob(os.path.join(path_to_image_folder , "*.png")))

    list_of_mask_paths = []
    for img_path in list_of_image_paths:

      _, filename = os.path.split(img_path)
      name , ext = os.path.splitext(filename)
      mask_name = name +labe_string + ext
      mask_path = os.path.join(path_to_mask_folder, mask_name)

      if os.path.exists(mask_path):
        list_of_mask_paths.append(mask_path)
      else:
        print(f"Warning: missing mask for {img_path}")

    image_paths_tf = tf.constant(list_of_image_paths, dtype=tf.string)
    mask_paths_tf  = tf.constant(list_of_mask_paths,  dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tf, mask_paths_tf))

    return dataset


##--------------------------------------------------------------------------------------------------------------------

import math
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images_masks(images, masks, pred_masks=None):
    """
    Plots a batch of (image, mask[, pred_mask]) triplets, laying them out
    in groups of 4 per row. If pred_masks is provided, each image row is:
    
      ┌──────────┬──────────┬──────────┬──────────┐
      │ Image 1  │ Image 2  │ Image 3  │ Image 4  │  <-- images
      ├──────────┼──────────┼──────────┼──────────┤
      │ Mask 1   │ Mask 2   │ Mask 3   │ Mask 4   │  <-- ground-truth masks
      ├──────────┼──────────┼──────────┼──────────┤
      │ Pred 1   │ Pred 2   │ Pred 3   │ Pred 4   │  <-- predicted masks
      └──────────┴──────────┴──────────┴──────────┘

    If pred_masks is None, only two rows per block (images + masks) are drawn.
    If your batch size is not a multiple of 4, the last row will have fewer than 4 entries.

    Args:
      images:     A Tensor of shape (B, H, W, 3) or a NumPy array, with pixel values in [0..255] or [0..1].
      masks:      A Tensor of shape (B, H, W) or (B, H, W, 1), integer class IDs.
      pred_masks: (Optional) A Tensor of shape (B, H, W) or (B, H, W, 1), your model’s predicted masks.

    Example:
      plot_images_masks(img_batch, mask_batch)
      plot_images_masks(img_batch, mask_batch, pred_mask_batch)
    """
    # Ensure eager numpy conversion is available
    images_np    = images.numpy()    if tf.is_tensor(images)    else images
    masks_np     = masks.numpy()     if tf.is_tensor(masks)     else masks
    has_preds    = (pred_masks is not None)
    pred_np      = pred_masks.numpy() if has_preds and tf.is_tensor(pred_masks) else pred_masks
    
    B = images_np.shape[0]                  # total number of examples in this batch
    cols = 4                                 # fix 4 columns per “block”
    blocks = math.ceil(B / cols)             # how many groups of 4 we need
    
    # Number of sub‐rows per block: 2 if no preds, else 3
    sub_rows = 3 if has_preds else 2
    
    # Total rows = blocks * sub_rows
    total_rows = blocks * sub_rows
    
    # Set up figure size: each plot approx. 4×4 inches
    plt.figure(figsize=(cols * 4, total_rows * 3))

    for idx in range(B):
        block_idx = idx // cols                     # which group of 4 we're in
        col_idx   = idx % cols                      # position within that group (0..3)

        # Row offset for this block:
        #   for block_idx=0, row_offset=0
        #   for block_idx=1, row_offset = sub_rows
        #   for block_idx=2, row_offset = 2*sub_rows, etc.
        row_offset = block_idx * sub_rows

        # 1) Plot image at (row_offset, col_idx)
        ax_image = plt.subplot(total_rows, cols, (row_offset * cols) + col_idx + 1)
        ax_image.imshow(images_np[idx] / 255.0 if images_np[idx].max() > 1.0 else images_np[idx])
        ax_image.set_title(f"Image {idx+1}")
        ax_image.axis("off")

        # 2) Plot ground‐truth mask at (row_offset + 1, col_idx)
        ax_mask = plt.subplot(total_rows, cols, ((row_offset + 1) * cols) + col_idx + 1)
        mask_i = masks_np[idx]
        if mask_i.ndim == 3 and mask_i.shape[-1] == 1:
            mask_i = mask_i[..., 0]
        ax_mask.imshow(mask_i, cmap="nipy_spectral")
        ax_mask.set_title(f"Mask {idx+1}")
        ax_mask.axis("off")

        # 3) If pred_masks given, plot at (row_offset + 2, col_idx)
        if has_preds:
            ax_pred = plt.subplot(total_rows, cols, ((row_offset + 2) * cols) + col_idx + 1)
            pm = pred_np[idx]
            if pm.ndim == 3 and pm.shape[-1] == 1:
                pm = pm[..., 0]
            ax_pred.imshow(pm, cmap="nipy_spectral")
            ax_pred.set_title(f"Pred {idx+1}")
            ax_pred.axis("off")

    plt.tight_layout()
    plt.show()

##--------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import tensorflow as tf

def preprocess_rgb_mask(mask_path: str) -> tf.Tensor:
    """
    Reads an RGB‐encoded segmentation PNG from `mask_path` and returns a
    single‐channel (H, W) tensor of integer class IDs.

    This function uses a tf.lookup.StaticHashTable to map packed RGB colors
    → class IDs (0–20, or 255 for “ignore”). Any color not in VOC’s list
    becomes 0 (background).

    Args:
      mask_path: Filesystem path (string) to a PNG where each pixel’s (R,G,B)
                 encodes a VOC class color.

    Returns:
      A tf.Tensor of shape (H, W), dtype=tf.uint8, where each value ∈ {0..20,255}.
    """
    # 1) Define VOC’s RGB→ID mapping as two 1D tensors of “packed” keys and IDs.
    #    We pack each (R,G,B) into a single 32‐bit int: (R<<16) | (G<<8) | B.
    voc_colors = tf.constant([
        [0,   0,   0],    # background → 0
        [128, 0,   0],    # aeroplane → 1
        [0,   128, 0],    # bicycle   → 2
        [128, 128, 0],    # bird      → 3
        [0,   0,   128],  # boat      → 4
        [128, 0,   128],  # bottle    → 5
        [0,   128, 128],  # bus       → 6
        [128, 128, 128],  # car       → 7
        [64,  0,   0],    # cat       → 8
        [192, 0,   0],    # chair     → 9
        [64,  128, 0],    # cow       → 10
        [192, 128, 0],    # diningtable → 11
        [64,  0,   128],  # dog       → 12
        [192, 0,   128],  # horse     → 13
        [64,  128, 128],  # motorbike → 14
        [192, 128, 128],  # person    → 15
        [0,   64,  0],    # pottedplant → 16
        [128, 64,  0],    # sheep     → 17
        [0,   192, 0],    # sofa      → 18
        [128, 192, 0],    # train     → 19
        [0,   64,  128],  # tvmonitor → 20
        [224, 224, 192],  # “ignore” color → 255
    ], dtype=tf.uint8)  # shape = (22, 3)

    # The corresponding class IDs for each color above:
    #   [0→0, 128,0,0→1, 0,128,0→2, … 0,64,128→20, 224,224,192→255]
    class_ids = tf.constant([
        0,   # background
        1,   # aeroplane
        2,   # bicycle
        3,   # bird
        4,   # boat
        5,   # bottle
        6,   # bus
        7,   # car
        8,   # cat
        9,   # chair
        10,  # cow
        11,  # diningtable
        12,  # dog
        13,  # horse
        14,  # motorbike
        15,  # person
        16,  # pottedplant
        17,  # sheep
        18,  # sofa
        19,  # train
        20,  # tvmonitor
        255  # ignore
    ], dtype=tf.uint8)  # shape = (22,)

    # 2) Pack each (R,G,B) color into a single int32 key:  key = (R << 16) | (G << 8) | B
    #    We’ll build a StaticHashTable from these packed keys → class IDs.
    rgb_flat = tf.reshape(voc_colors, [-1, 3])      # shape=(22,3)
    packed_keys = (tf.cast(rgb_flat[:, 0], tf.int32) << 16) | \
                  (tf.cast(rgb_flat[:, 1], tf.int32) << 8)  | \
                  (tf.cast(rgb_flat[:, 2], tf.int32))       # shape = (22,)

    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=packed_keys,
        values=tf.cast(class_ids, tf.int32),
        key_dtype=tf.int32,
        value_dtype=tf.int32
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=0)
    # default_value=0 ensures any unknown color → background (ID=0).

    # 3) Read and decode the PNG mask (H, W, 3), dtype=uint8
    file_bytes = tf.io.read_file(mask_path)
    rgb_mask = tf.io.decode_png(file_bytes, channels=3)  # shape=(H, W, 3), dtype=uint8

    # 4) Pack every pixel in rgb_mask into its int32 key
    r = tf.cast(rgb_mask[:, :, 0], tf.int32)
    g = tf.cast(rgb_mask[:, :, 1], tf.int32)
    b = tf.cast(rgb_mask[:, :, 2], tf.int32)
    pixel_keys = (r << 16) | (g << 8) | b  # shape = (H, W), dtype=int32

    # 5) Lookup each key in the table → single‐channel class ID map
    id_mask = table.lookup(pixel_keys)     # shape = (H, W), dtype=int32

    # 6) Cast to uint8 before returning
    return tf.cast(id_mask, tf.uint8)

##--------------------------------------------------------------------------------------------------------------------
def preprocess_rgb_mask_with_csv(
    mask_path: str,
    csv_path: str,
    r_col: str = "r",
    g_col: str = "g",
    b_col: str = "b",
    id_col: str = None,
    ignore_color: tuple[int, int, int] = None,
    ignore_id: int = 255,
) -> tf.Tensor:
    """
    Reads a color‐coded PNG segmentation mask from `mask_path` and a CSV file that
    describes how to map each RGB triple → integer class ID. Returns a single‐channel
    (H, W) mask tensor of dtype uint8, where each pixel’s value is the class ID.

    CSV requirements:
      • Must contain three columns with names matching `r_col`, `g_col`, `b_col` (all integers 0–255).
      • If `id_col` is provided, that column should be integer class IDs.
      • If `id_col` is None, then each row’s index in the CSV (0, 1, 2, …) becomes its class ID.

    Optional `ignore_color`:
      • A 3-tuple (R, G, B) that, if encountered in the mask, is mapped to `ignore_id` (default 255).
      • If `ignore_color` is None, no explicit ignore color is added.

    Background or any RGB not in CSV → ID = 0 (unless overridden by an explicit ignore_color).

    Args:
      mask_path: Path to a PNG file where each pixel’s (R, G, B) represents a class color.
      csv_path:   Path to a CSV file defining the color→class mapping.
      r_col:      Column name in CSV for the Red channel (default “r”).
      g_col:      Column name in CSV for the Green channel (default “g”).
      b_col:      Column name in CSV for the Blue channel (default “b”).
      id_col:     (Optional) Column name for integer class IDs. If None, use row index as ID.
      ignore_color:  (Optional) 3-tuple (R, G, B) that maps to `ignore_id`. Example: (224, 224, 192).
      ignore_id:     Integer ID to assign to `ignore_color` (default 255).

    Returns:
      A tf.Tensor of shape (H, W), dtype=tf.uint8, where each pixel’s value is the mapped class ID.
    """
    # 1) Load CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    # 2) Determine class IDs
    if id_col is not None:
        # If CSV provides an explicit column of class IDs, use it
        class_ids = df[id_col].astype(np.int32).to_numpy()
    else:
        # Otherwise, use the dataframe’s row index as the class ID for each entry
        class_ids = df.index.to_numpy(dtype=np.int32)
    # Now: class_ids.shape = (N,), each is an integer ID >= 0

    # 3) Extract RGB colors from CSV
    colors = df[[r_col, g_col, b_col]].astype(np.int32).to_numpy()  
    # colors.shape = (N, 3), dtype=int32

    # 4) Pack each (R,G,B) into a single 32-bit integer key: key = (R << 16) | (G << 8) | B
    packed_keys = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]  
    # packed_keys.shape = (N,), dtype=int32

    # 5) If an ignore_color is specified, add it to the mapping
    if ignore_color is not None:
        # Compute its packed key
        r_ig, g_ig, b_ig = ignore_color
        packed_ignore = (int(r_ig) << 16) | (int(g_ig) << 8) | int(b_ig)
        # Append to our arrays
        packed_keys = np.concatenate([packed_keys, np.array([packed_ignore], dtype=np.int32)], axis=0)
        class_ids = np.concatenate([class_ids, np.array([ignore_id], dtype=np.int32)], axis=0)

    # 6) Build a TensorFlow StaticHashTable from packed_keys → class_ids
    keys_tensor = tf.constant(packed_keys, dtype=tf.int32)      # shape = (N_or_N+1,)
    values_tensor = tf.constant(class_ids, dtype=tf.int32)      # shape = (N_or_N+1,)
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=keys_tensor,
        values=values_tensor,
        key_dtype=tf.int32,
        value_dtype=tf.int32,
    )
    # Default to 0 for any color not in CSV (treat as background)
    lookup_table = tf.lookup.StaticHashTable(initializer, default_value=0)

    # 7) Read and decode the PNG mask from disk (H, W, 3), dtype=uint8
    file_bytes = tf.io.read_file(mask_path)
    rgb_mask = tf.io.decode_png(file_bytes, channels=3)  
    # rgb_mask.shape = (H, W, 3), dtype=uint8

    # 8) Compute the packed key for every pixel: (R << 16) | (G << 8) | B
    r_channel = tf.cast(rgb_mask[..., 0], tf.int32)
    g_channel = tf.cast(rgb_mask[..., 1], tf.int32)
    b_channel = tf.cast(rgb_mask[..., 2], tf.int32)
    pixel_keys = (r_channel << 16) | (g_channel << 8) | b_channel  # shape = (H, W)

    # 9) Look up each pixel_key in the table → single‐channel ID mask
    id_mask = lookup_table.lookup(pixel_keys)  # shape = (H, W), dtype=int32

    # 10) Cast to uint8 (or leave as int32 if you expect IDs > 255)
    return tf.cast(id_mask, tf.uint8)

##--------------------------------------------------------------------------------------------------------------------

#fucntion for preprocessing image and mask it take datset of image_path and mask_path (tf.data.Dataset.from_tensor_slices((list_of_image_paths, list_of_mask_paths))) 
# and you can you ues it by maping the dataset
## !! you will need to cheange the csv_path and the col name if need it
def load_and_preprocess_image_mask(image_path, mask_path):
    # ----- IMAGE SIDE -----
    #  a) Read file bytes, decode to uint8 tensor, resize & normalize
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(raw, channels=3)      # (H, W, 3), uint8
    img = tf.image.resize(img, [256, 256])       # choose your IMG_SIZE
    img = tf.cast(img, tf.float32) / 255.0       # now in [0,1]

    # ----- MASK SIDE -----
    #  b) Convert from color‐coded PNG → single‐channel ID mask
    id_mask = preprocess_rgb_mask_with_csv(
        mask_path=mask_path,
        csv_path="/path/to/your/colormap.csv",
        r_col="# r",  # exactly match your CSV header
        g_col="# g",
        b_col="# b",
        id_col=None,                  # if your CSV has no explicit “ID” column
        ignore_color=(224, 224, 192),
        ignore_id=255
    )
    #  c) Resize the ID mask using nearest‐neighbor (so IDs don’t get interpolated)
    id_mask = tf.image.resize(id_mask[..., None], [256, 256], method="nearest")
    id_mask = tf.squeeze(id_mask, axis=-1)  # shape = (256, 256), dtype=uint8

    return img, id_mask
