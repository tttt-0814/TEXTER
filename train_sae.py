import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import TensorDataset

from src.utils.load_models import setup_model
from src.overcomplete.sae import TopKSAE, trainer_sae
from src.overcomplete.sae.train_utils import (
    save_features,
    load_features,
    save_logs_to_file,
    save_args_to_file,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Auto Encoder with ResNet50 features"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar10", "imagenet"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "vit", "dino_vits8", "dino_resnet50"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/yamauchi/WorkSpace/mnt/nfs/yamauchi/Data/ImageNet",
        # default="/home/yamauchi/WorkSpace/mnt/nfs/yamauchi/Data/CIFAR10",
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--feature_extract_batch_size",
        type=int,
        default=128,
        help="Batch size for feature extraction",
    )
    # Training arguments
    parser.add_argument(
        "--sae_batch_size", type=int, default=1024, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    # parser.add_argument("--top_k", type=int, default=32, help="Top-k for SAE")
    parser.add_argument(
        "--num_multi_dict", type=int, default=8, help="Number of multi-dictionary"
    )

    # Device and output
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu). If None, auto-detect",
    )
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sae_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Force feature extraction even if cached features exist",
    )

    return parser.parse_args()


def extract_features(model, dataloader, device):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(
            tqdm(dataloader, desc="Extracting features")
        ):
            data = data.to(device)

            # Apply normalization if model has normalizer
            if model.has_normalizer:
                data = model.get_normalizer(data)

            # Extract features using forward_features
            batch_features = model.forward_features(data)
            batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten

            features.append(batch_features.cpu())
            labels.append(target)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


def extract_or_load_features(model, dataloader, device, features_path, force_extract):
    """
    Extract features from ResNet50 model and save them to a file,
    or load them from a file if they exist and force_extract is False.
    """
    if features_path.exists() and not force_extract:
        print(f"Loading features from {features_path}")
        features, labels = load_features(features_path)
        print(f"Features loaded. Shape: {features.shape}")
    else:
        print(f"Extracting features and saving to {features_path}")
        features, labels = extract_features(model, dataloader, device)
        save_features(features, labels, features_path)
        print(f"Features extracted and saved. Shape: {features.shape}")

    return features, labels


def main():
    """Main training function."""
    args = parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create features directory
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard logging
    log_dir = output_dir / "tensorboard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Directory structure:")
    print(f"  Output dir: {output_dir}")
    print(f"  Features dir: {features_dir}")
    print(f"  TensorBoard logs: {log_dir}")

    # Save arguments
    args_path = output_dir / "args.json"
    save_args_to_file(args, args_path)

    model = setup_model(args.model_name, pretrained=True, device=device)
    print(f"Model {args.model_name} setup complete")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # Setup transforms
    if args.dataset == "cifar10":
        # Load CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=True, transform=transform
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=transform
        )

    elif args.dataset == "imagenet":
        # Load ImageNet dataset
        train_dataset = torchvision.datasets.ImageNet(
            root=args.data_root, split="train", transform=transform
        )

        test_dataset = torchvision.datasets.ImageNet(
            root=args.data_root, split="val", transform=transform
        )

    print(f"Dataset: {args.dataset}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.feature_extract_batch_size,
        shuffle=False,  # No need to shuffle for feature extraction
        num_workers=8,
    )

    # Extract or load features from training data
    train_features_path = features_dir / f"{args.dataset}_train_features.pth"
    train_features, train_labels = extract_or_load_features(
        model, train_loader, device, train_features_path, args.force_extract
    )
    print(f"Training features shape: {train_features.shape}")

    # Initialize SAE
    input_dim = train_features.shape[1]  # Feature dimension from ResNet50
    nb_concepts = input_dim * args.num_multi_dict
    top_k = nb_concepts // 10

    sae = TopKSAE(
        input_dim,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=args.device,
    )

    dataloader = DataLoader(
        TensorDataset(train_features),
        batch_size=args.sae_batch_size,
        shuffle=True,
        num_workers=8,
    )
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    def criterion(x, x_hat, pre_codes, codes, dictionary):
        mse = (x - x_hat).square().mean()
        return mse

    logs = trainer_sae(
        sae,
        dataloader,
        criterion,
        optimizer,
        nb_epochs=args.epochs,
        device=args.device,
        writer=writer,
    )

    # Save trained model
    model_path = output_dir / "sae_model.pth"
    torch.save(
        {
            "model_state_dict": sae.state_dict(),
            "args": args,
            "input_dim": input_dim,
            "nb_concepts": nb_concepts,
            "top_k": top_k,
        },
        model_path,
    )

    print(f"Training complete! Model saved to {model_path}")

    # Save training logs
    logs_path = output_dir / "logs.txt"
    save_logs_to_file(logs, logs_path)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
