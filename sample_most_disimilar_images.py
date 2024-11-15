#!/usr/bin/env python3

import argparse
import os
import shutil

import numpy as np
import torch

from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Select the K most dissimilar images from a set, ensuring no more than N consecutive frames are skipped."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="selected_images",
        help="Directory to save the selected images.",
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Total number of images to select.",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Maximum number of consecutive frames to skip between selected frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to run the model on (e.g., "cpu", "cuda"). Defaults to GPU if available.',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dino",
        choices=["clip", "dino"],
        help='Model to use for feature extraction ("clip" or "dino").',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for feature extraction.",
    )
    args = parser.parse_args()
    return args


def load_and_preprocess_images(image_paths, preprocess, device):
    features = []
    valid_image_paths = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            features.append(input_tensor)
            valid_image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    if not features:
        raise ValueError("No valid images were found.")
    return torch.cat(features, dim=0), valid_image_paths


def extract_features(model_name, model, images, batch_size: int):
    all_features = []

    # Split images into batches
    images_batched = torch.split(images, batch_size)

    with torch.no_grad():
        for image_batch in tqdm(images_batched, desc="Extracting features (batches)"):
            if model_name == "clip":
                batch_features = model.encode_image(image_batch)
            elif model_name == "dino":
                batch_features = model(image_batch)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            # Normalize features
            batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)

            # Append batch features
            all_features.append(batch_features.cpu().numpy())

    # Concatenate all batch features
    return np.concatenate(all_features, axis=0)


def select_most_dissimilar_images(
    features, initial_selected_indices, num_to_select, available_indices
):
    distance_matrix = cdist(features, features, metric="cosine")
    selected_indices = list(initial_selected_indices)

    remaining_indices = set(available_indices) - set(selected_indices)

    for _ in tqdm(range(num_to_select), "Selecting most dissimilar images"):
        max_min_distance = -1
        next_index = -1

        for idx in remaining_indices:
            # Compute the minimal distance to the selected images
            min_distance = np.min(distance_matrix[idx, selected_indices])

            # Select the image with the maximal minimal distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_index = idx

        if next_index == -1:
            print("No more images to select.")
            break

        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    return selected_indices


def main():
    args = parse_arguments()

    # Automatically select device if not specified
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Get all image file paths
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    image_paths = [
        os.path.join(args.image_dir, fname)
        for fname in sorted(os.listdir(args.image_dir))
        if fname.lower().endswith(image_extensions)
    ]

    if len(image_paths) < args.K:
        print(
            f"Error: Number of images ({len(image_paths)}) is less than K ({args.K})."
        )
        return

    # Load the selected model
    if args.model == "clip":
        import clip

        model, preprocess = clip.load("ViT-B/32", device=device)
    elif args.model == "dino":
        # Install timm if not already installed
        try:
            import timm
        except ImportError:
            print("Installing timm library...")
            os.system("pip install timm")
            import timm

        model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
        model.eval()
        model.to(device)

        # Define DINO preprocessing
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize(248),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),  # ImageNet means
                    std=(0.229, 0.224, 0.225),  # ImageNet stds
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Load and preprocess images
    images, valid_image_paths = load_and_preprocess_images(
        image_paths, preprocess, device
    )

    # Sample uniformly ensuring no more than N frames are skipped
    first_round_indices = list(range(0, len(valid_image_paths), args.N + 1))
    remaining_indices = set(range(len(valid_image_paths))) - set(first_round_indices)
    print(f"Attempting to select {len(first_round_indices)} images uniformly.")

    # Extract features
    features = extract_features(args.model, model, images, args.batch_size)

    if len(first_round_indices) >= args.K:
        selected_indices = first_round_indices[: args.K]
        num_first_round = len(selected_indices)
        num_second_round = 0
    else:
        num_first_round = len(first_round_indices)
        num_second_round = args.K - num_first_round
        selected_indices = select_most_dissimilar_images(
            features, first_round_indices, num_second_round, remaining_indices
        )

    num_total_selected = len(selected_indices)
    num_second_round = num_total_selected - num_first_round

    print(f"Number of frames sampled in first round (uniform): {num_first_round}")
    print(
        f"Number of frames sampled in second round (feature dissimilarity): {num_second_round}"
    )

    # Get the selected images
    selected_images = [valid_image_paths[idx] for idx in selected_indices]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy selected images to the output directory
    for img_path in selected_images:
        shutil.copy(img_path, args.output_dir)

    print(f"Selected {args.K} images using {args.model.upper()}.")
    print(f"\nSelected images have been saved to '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()
