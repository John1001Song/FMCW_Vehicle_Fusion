import torch

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected. Please ensure that CUDA is installed correctly.")

if __name__ == "__main__":
    main()
